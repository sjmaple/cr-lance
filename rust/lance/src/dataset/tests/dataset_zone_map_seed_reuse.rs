// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Zone map builds reuse the write seeds fragments already carry, whatever
//! record size those seeds were written with, and scan only the fragments
//! without a usable seed. Every claim about what was reused or scanned is
//! taken from the build itself, and every query is compared with a plain
//! scan by row identity.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use arrow_array::{
    Array, Float64Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::utils::tempfile::TempStrDir;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_index::IndexType;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::pbold;
use lance_index::scalar::seed::SEED_META_KEY_PREFIX;
use lance_index::scalar::zonemap::ZoneMapIndex;
use lance_index::scalar::{BuiltinIndexType, SargableQuery, ScalarIndexParams};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_io::utils::tracking_store::IoStats;
use lance_table::format::{DataFile, IndexMetadata};
use roaring::RoaringBitmap;

use crate::Dataset;
use crate::dataset::WriteParams;
use crate::index::scalar::zonemap::{SEED_REUSE_REPORTS, SeedReuseReport};
use crate::index::{CreateIndexBuilder, DatasetIndexExt, DatasetIndexInternalExt};

const ROWS_PER_FILE: usize = 20000;
/// Rows per written fragment.
const N: i32 = ROWS_PER_FILE as i32;

fn index_name(column: &str) -> String {
    format!("{column}_zones")
}

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("val", DataType::Int64, true),
        ArrowField::new("name", DataType::Utf8, true),
        ArrowField::new("x", DataType::Float64, true),
    ]))
}

/// `val` is null every fifth row and otherwise a scrambled multiple of the
/// id, so zones overlap and most predicates prune some but not all of them.
fn val(i: i32) -> Option<i64> {
    (i % 5 != 0).then(|| i64::from(i) * 7 % 1013 + i64::from(i) / 100 * 1000)
}

fn name(i: i32) -> Option<String> {
    (i % 11 != 0).then(|| format!("n{:05}", (i * 37) % 20000))
}

fn x(i: i32) -> Option<f64> {
    match i % 13 {
        0 => None,
        1 => Some(f64::NAN),
        2 => Some(-f64::NAN),
        3 => Some(f64::INFINITY),
        4 => Some(f64::NEG_INFINITY),
        _ => Some(f64::from(i % 97) - 48.0),
    }
}

fn rows(ids: std::ops::Range<i32>) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from_iter_values(ids.clone())),
            Arc::new(Int64Array::from_iter(ids.clone().map(val))),
            Arc::new(StringArray::from_iter(ids.clone().map(name))),
            Arc::new(Float64Array::from_iter(ids.map(x))),
        ],
    )
    .unwrap()
}

fn zone_map_params(
    rows_per_zone: u64,
    use_seeds: bool,
    store_null_bitmap: bool,
) -> ScalarIndexParams {
    ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(&serde_json::json!({
        "rows_per_zone": rows_per_zone,
        "use_seeds": use_seeds,
        "store_null_bitmap": store_null_bitmap,
    }))
}

async fn write(uri: &str, ids: std::ops::Range<i32>) -> Dataset {
    Dataset::write(
        RecordBatchIterator::new([Ok(rows(ids))], schema()),
        uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FILE,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

async fn append(dataset: &mut Dataset, ids: std::ops::Range<i32>) {
    dataset
        .append(RecordBatchIterator::new([Ok(rows(ids))], schema()), None)
        .await
        .unwrap();
}

/// Create (or replace) the zone map on `column`, over `fragments` when given.
async fn create_index(
    dataset: &mut Dataset,
    column: &str,
    params: &ScalarIndexParams,
    fragments: Option<Vec<u32>>,
) {
    let mut builder = CreateIndexBuilder::new(dataset, &[column], IndexType::ZoneMap, params)
        .name(index_name(column))
        .replace(true);
    if let Some(fragments) = fragments {
        builder = builder.fragments(fragments);
    }
    builder.await.unwrap();
}

/// Run `build` and return the seed reuse reports it produced.
async fn with_reports<F, T>(build: F) -> (T, Vec<SeedReuseReport>)
where
    F: std::future::Future<Output = T>,
{
    SEED_REUSE_REPORTS
        .scope(Mutex::new(Vec::new()), async {
            let out = build.await;
            let reports = SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone());
            (out, reports)
        })
        .await
}

async fn index_metadata(dataset: &Dataset, column: &str) -> Vec<IndexMetadata> {
    dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|index| index.name == index_name(column))
        .cloned()
        .collect()
}

fn details(segment: &IndexMetadata) -> pbold::ZoneMapIndexDetails {
    segment
        .index_details
        .as_ref()
        .unwrap()
        .to_msg::<pbold::ZoneMapIndexDetails>()
        .unwrap()
}

fn coverage(segment: &IndexMetadata) -> RoaringBitmap {
    segment.fragment_bitmap.clone().unwrap()
}

/// Ids matching `filter`, through the index or through a plain scan. With
/// `expect_index` the plan must show the zone map being consulted.
async fn ids_matching(
    dataset: &Dataset,
    filter: &str,
    use_index: bool,
    expect_index: bool,
) -> BTreeSet<i32> {
    let mut scanner = dataset.scan();
    scanner
        .filter(filter)
        .unwrap()
        .project(&["id"])
        .unwrap()
        .use_scalar_index(use_index);
    if use_index && expect_index {
        let plan = scanner.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery"),
            "{filter}: the zone map was not used:\n{plan}"
        );
    }
    let batch = scanner.try_into_batch().await.unwrap();
    batch["id"]
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .values()
        .iter()
        .copied()
        .collect()
}

const VAL_FILTERS: &[&str] = &[
    "val > 500",
    "val < 2300",
    "val >= 7000 AND val <= 7300",
    "val = 707",
    "val IN (14, 20000, 8191)",
    "val IS NULL",
    "val IS NOT NULL",
    "val BETWEEN 100 AND 110",
];

/// Every filter yields the same ids with and without the index. The index
/// must appear in the plan, except for `IS NOT NULL` on an index without a
/// null bitmap, which the planner leaves to the ordinary filter.
async fn assert_index_matches_scan(dataset: &Dataset, filters: &[&str]) {
    assert_index_matches_scan_with(dataset, filters, true).await;
}

async fn assert_index_matches_scan_with(
    dataset: &Dataset,
    filters: &[&str],
    has_null_bitmap: bool,
) {
    for filter in filters {
        let expect_index = has_null_bitmap || !filter.contains("IS NOT NULL");
        let indexed = ids_matching(dataset, filter, true, expect_index).await;
        let scanned = ids_matching(dataset, filter, false, false).await;
        assert_eq!(indexed, scanned, "{filter}");
    }
}

fn io(dataset: &Dataset) -> IoStats {
    dataset.object_store.io_stats_incremental()
}

async fn open_data_file(dataset: &Dataset, file: &DataFile) -> FileReader {
    let path = dataset
        .data_file_dir(file)
        .unwrap()
        .join(file.path.as_str());
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let file_scheduler = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await
        .unwrap();
    FileReader::try_open(
        file_scheduler,
        None,
        Default::default(),
        &dataset.metadata_cache.file_metadata_cache(&path),
        FileReaderOptions::default(),
    )
    .await
    .unwrap()
}

async fn seed_locator(dataset: &Dataset, fragment_id: u64, column: &str) -> Option<String> {
    let fragment = dataset
        .fragments()
        .iter()
        .find(|f| f.id == fragment_id)
        .unwrap()
        .clone();
    let reader = open_data_file(dataset, fragment.files.last().unwrap()).await;
    reader
        .metadata()
        .file_schema
        .metadata
        .get(&format!("{SEED_META_KEY_PREFIX}{column}"))
        .cloned()
}

/// Bytes read from the dataset's data files, split into the column data
/// page section and everything else (footer, column metadata and global
/// buffers such as seeds). Reads of other objects are ignored.
/// Per data file: (column page bytes, footer/metadata/seed bytes) read.
async fn classify_data_file_reads(dataset: &Dataset, io: &IoStats) -> HashMap<String, (u64, u64)> {
    // Pages and global buffers (seeds) share the region before the column
    // metadata; the global buffers are carved out of it.
    let mut layout = HashMap::new();
    for fragment in dataset.fragments().iter() {
        for file in &fragment.files {
            let path = dataset
                .data_file_dir(file)
                .unwrap()
                .join(file.path.as_str());
            let metadata = open_data_file(dataset, file).await.metadata().clone();
            let data_end = metadata.num_data_bytes + metadata.num_global_buffer_bytes;
            let buffers: Vec<std::ops::Range<u64>> = metadata
                .file_buffers
                .iter()
                .map(|buffer| buffer.position..buffer.position + buffer.size)
                .collect();
            layout.insert(path.to_string(), (data_end, buffers));
        }
    }
    fn overlap(a: &std::ops::Range<u64>, b: &std::ops::Range<u64>) -> u64 {
        a.end.min(b.end).saturating_sub(a.start.max(b.start))
    }
    let mut per_file: HashMap<String, (u64, u64)> = HashMap::new();
    for request in &io.requests {
        let Some(range) = &request.range else {
            continue;
        };
        let Some((data_end, buffers)) = layout.get(&request.path.to_string()) else {
            continue;
        };
        let in_region = overlap(range, &(0..*data_end));
        let in_buffers: u64 = buffers.iter().map(|buffer| overlap(range, buffer)).sum();
        let in_pages = in_region - in_buffers;
        let entry = per_file.entry(request.path.to_string()).or_default();
        entry.0 += in_pages;
        entry.1 += (range.end - range.start) - in_pages;
    }
    per_file
}

/// Column page bytes read from each fragment's serving file for `column`.
fn pages_per_fragment(
    dataset: &Dataset,
    column: &str,
    per_file: &HashMap<String, (u64, u64)>,
) -> Vec<(u64, u64)> {
    let field = dataset.schema().field(column).unwrap().id;
    dataset
        .fragments()
        .iter()
        .map(|fragment| {
            let file = fragment
                .files
                .iter()
                .rev()
                .find(|file| file.fields.contains(&field))
                .unwrap();
            let path = dataset
                .data_file_dir(file)
                .unwrap()
                .join(file.path.as_str())
                .to_string();
            (fragment.id, per_file.get(&path).map_or(0, |v| v.0))
        })
        .collect()
}

fn totals(per_file: &HashMap<String, (u64, u64)>) -> (u64, u64) {
    per_file
        .values()
        .fold((0, 0), |acc, v| (acc.0 + v.0, acc.1 + v.1))
}

/// Seeds written for an 8-row index fold into one 20000-row zone per fragment
/// at creation, without reading those fragments' values.
#[tokio::test]
async fn test_create_index_packs_smaller_seed_records_into_large_zones() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(&mut dataset, "val", &zone_map_params(64, true, true), None).await;
    for k in 1..6 {
        append(&mut dataset, k * N..(k + 1) * N).await;
    }
    assert_eq!(
        seed_locator(&dataset, 3, "val")
            .await
            .as_deref()
            .unwrap()
            .split(':')
            .nth(1),
        Some("64")
    );
    dataset.drop_index(&index_name("val")).await.unwrap();

    io(&dataset);
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(20000, true, true),
        None,
    ))
    .await;
    let reuse_io = io(&dataset);
    let reuse_reads = classify_data_file_reads(&dataset, &reuse_io).await;
    let (reuse_pages, reuse_other) = totals(&reuse_reads);
    // Per fragment: a reused fragment's file contributes no column page
    // reads. Opening a file prefetches a fixed-size tail that can overlap
    // the end of the last page by under 4 KiB; that prefetch is the only
    // page-region byte a reused fragment may show.
    for (fragment, pages) in pages_per_fragment(&dataset, "val", &reuse_reads) {
        if fragment == 0 {
            assert!(
                pages > 4096,
                "the unseeded fragment is scanned: {pages} page bytes"
            );
        } else {
            assert!(
                pages < 4096,
                "fragment {fragment} was reused but {pages} page bytes were read"
            );
        }
    }
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].reused_fragments, vec![1, 2, 3, 4, 5]);
    assert_eq!(reports[0].reused_records, 5 * (N as usize / 64 + 1));
    assert_eq!(reports[0].zones_from_seeds, 5);
    assert_eq!(reports[0].scanned_fragments, vec![0]);
    // Raw seed buffers are owned one fragment at a time: the most ever
    // owned at once is one seed, far below the five harvested in total.
    assert!(reports[0].max_live_seed_bytes > 0);
    assert!(
        reports[0].max_live_seed_bytes * 4 <= reports[0].total_seed_bytes,
        "live {} of {} total seed bytes",
        reports[0].max_live_seed_bytes,
        reports[0].total_seed_bytes
    );

    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(details(&segments[0]).rows_per_zone, Some(20000));
    assert_eq!(details(&segments[0]).has_null_bitmap, Some(true));
    assert_eq!(coverage(&segments[0]), RoaringBitmap::from_iter(0..6u32));
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;

    // Control: the same table written without any index has no seeds, so
    // the same build scans every fragment.
    let control_dir = TempStrDir::default();
    let mut control = write(control_dir.as_str(), 0..N).await;
    for k in 1..6 {
        append(&mut control, k * N..(k + 1) * N).await;
    }
    io(&control);
    let (_, reports) = with_reports(create_index(
        &mut control,
        "val",
        &zone_map_params(20000, true, true),
        None,
    ))
    .await;
    let control_io = io(&control);
    let (control_pages, control_other) =
        totals(&classify_data_file_reads(&control, &control_io).await);
    assert!(reports[0].reused_fragments.is_empty());
    assert_eq!(reports[0].scanned_fragments, vec![0, 1, 2, 3, 4, 5]);
    // Only the one scanned fragment's column pages are read; the five reused
    // fragments cost footer and seed bytes only.
    assert!(
        reuse_pages * 4 <= control_pages,
        "column pages: reuse read {reuse_pages} bytes, scanning all six read {control_pages}"
    );
    log::info!(
        "SEED_REUSE_IO reuse: pages={reuse_pages} footer_and_seeds={reuse_other} total={}; \
         scan_all: pages={control_pages} footer_and_seeds={control_other} total={}",
        reuse_io.read_bytes,
        control_io.read_bytes
    );
    assert_index_matches_scan(&control, VAL_FILTERS).await;
}

/// An update folds seeds of two different record sizes into a 20000-row
/// index and keeps the exact null positions they carry.
#[tokio::test]
async fn test_update_reuses_seeds_of_any_record_size() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(&mut dataset, "val", &zone_map_params(8, true, true), None).await;
    append(&mut dataset, N..2 * N).await;
    append(&mut dataset, 2 * N..3 * N).await;
    // Replace the index over fragment 0 only: fragments 1 and 2 keep their
    // 8-row seeds and are unindexed for the new index.
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(20000, true, true),
        Some(vec![0]),
    )
    .await;
    append(&mut dataset, 3 * N..4 * N).await;
    assert_eq!(
        seed_locator(&dataset, 1, "val")
            .await
            .as_deref()
            .unwrap()
            .split(':')
            .nth(1),
        Some("8")
    );
    assert_eq!(
        seed_locator(&dataset, 3, "val")
            .await
            .as_deref()
            .unwrap()
            .split(':')
            .nth(1),
        Some("8192")
    );

    let (_, reports) = with_reports(async {
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
    })
    .await;
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].reused_fragments, vec![1, 2, 3]);
    assert_eq!(reports[0].reused_records, 2 * (N as usize / 8) + 3);
    assert_eq!(reports[0].zones_from_seeds, 3);
    assert!(reports[0].scanned_fragments.is_empty());

    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(segments.len(), 1);
    assert_eq!(coverage(&segments[0]), RoaringBitmap::from_iter(0..4u32));
    assert_eq!(details(&segments[0]).has_null_bitmap, Some(true));
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;
    let index = dataset
        .open_scalar_index("val", &segments[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let result = index
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(result.is_exact(), "IS NULL stays exact after a seed update");
    assert_eq!(
        index
            .as_any()
            .downcast_ref::<ZoneMapIndex>()
            .unwrap()
            .num_zones(),
        4
    );
}

/// A fragment without a seed is scanned while its seeded siblings are not.
#[tokio::test]
async fn test_update_scans_only_fragments_without_a_seed() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(&mut dataset, "val", &zone_map_params(16, true, true), None).await;
    append(&mut dataset, N..2 * N).await;
    dataset.drop_index(&index_name("val")).await.unwrap();
    append(&mut dataset, 2 * N..3 * N).await;
    assert!(seed_locator(&dataset, 2, "val").await.is_none());
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(16, true, true),
        Some(vec![0]),
    )
    .await;

    let (_, reports) = with_reports(async {
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
    })
    .await;
    assert_eq!(reports[0].reused_fragments, vec![1]);
    assert_eq!(reports[0].scanned_fragments, vec![2]);
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(coverage(&segments[0]), RoaringBitmap::from_iter(0..3u32));
    assert_eq!(details(&segments[0]).has_null_bitmap, Some(true));
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;
}

/// `store_null_bitmap: false` survives creation, updates and parameter
/// derivation, and null predicates stay correct through a recheck.
#[tokio::test]
async fn test_index_without_null_bitmap() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(100, true, false),
        None,
    )
    .await;
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(details(&segments[0]).has_null_bitmap, Some(false));
    assert_index_matches_scan_with(&dataset, VAL_FILTERS, false).await;

    // Appends reuse seeds and keep the setting.
    append(&mut dataset, N..2 * N).await;
    let (_, reports) = with_reports(async {
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
    })
    .await;
    assert_eq!(reports[0].reused_fragments, vec![1]);
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(details(&segments[0]).has_null_bitmap, Some(false));
    let index = dataset
        .open_scalar_index("val", &segments[0].uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let zone_map = index.as_any().downcast_ref::<ZoneMapIndex>().unwrap();
    assert!(!zone_map.store_null_bitmap());
    assert!(!zone_map.has_null_bitmap());
    let result = index
        .search(&SargableQuery::IsNull(), &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(!result.is_exact());
    let derived = index.derive_index_params().unwrap();
    assert!(
        derived
            .params
            .unwrap()
            .contains("\"store_null_bitmap\":false")
    );
    assert_index_matches_scan_with(&dataset, VAL_FILTERS, false).await;

    // A rebuild from derived parameters keeps the setting too.
    append(&mut dataset, 2 * N..3 * N).await;
    dataset
        .optimize_indices(&OptimizeOptions::new().num_indices_to_merge(Some(0)))
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::new().num_indices_to_merge(Some(2)))
        .await
        .unwrap();
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(segments.len(), 1);
    assert_eq!(details(&segments[0]).has_null_bitmap, Some(false));
    assert_eq!(coverage(&segments[0]), RoaringBitmap::from_iter(0..3u32));
    assert_index_matches_scan_with(&dataset, VAL_FILTERS, false).await;
}

/// Seeds describe physical rows. A fragment with deletions, including one
/// whose live rows are all null, is still summarized from its seed and the
/// index stays a superset of the live matches.
#[tokio::test]
async fn test_seed_reuse_with_deleted_rows() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(&mut dataset, "val", &zone_map_params(8, true, true), None).await;
    append(&mut dataset, N..2 * N).await;
    append(&mut dataset, 2 * N..3 * N).await;
    // Fragment 1: delete every third row, and every non-null row of its first
    // hundred ids
    // so the survivors of that range are all null.
    dataset.delete("id % 3 = 0").await.unwrap();
    dataset
        .delete(&format!(
            "id >= {N} AND id < {} AND val IS NOT NULL",
            N + 100
        ))
        .await
        .unwrap();
    // Fragment 2: delete everything, which removes the fragment.
    dataset.delete(&format!("id >= {}", 2 * N)).await.unwrap();
    assert_eq!(dataset.fragments().len(), 2);
    let physical: u64 = dataset.fragments()[1].physical_rows.unwrap() as u64;
    assert_eq!(physical, N as u64);
    dataset.drop_index(&index_name("val")).await.unwrap();

    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(20000, true, true),
        None,
    ))
    .await;
    assert_eq!(reports[0].reused_fragments, vec![1]);
    assert_eq!(reports[0].scanned_fragments, vec![0]);
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(coverage(&segments[0]), RoaringBitmap::from_iter([0u32, 1]));
    assert_index_matches_scan(
        &dataset,
        &[
            "val IS NULL",
            "val IS NOT NULL",
            "val > 0",
            "val < 0",
            &format!("id >= {N} AND id < {} AND val IS NOT NULL", N + 100),
            "val IN (14, 20000, 8191)",
        ],
    )
    .await;
}

/// Creating over a subset of fragments harvests and covers only that subset.
#[tokio::test]
async fn test_subset_creation_stays_within_the_requested_fragments() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(&mut dataset, "val", &zone_map_params(8, true, true), None).await;
    append(&mut dataset, N..2 * N).await;
    append(&mut dataset, 2 * N..3 * N).await;
    dataset.drop_index(&index_name("val")).await.unwrap();
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(20000, true, true),
        Some(vec![1, 2]),
    ))
    .await;
    assert_eq!(reports[0].reused_fragments, vec![1, 2]);
    assert!(reports[0].scanned_fragments.is_empty());
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(coverage(&segments[0]), RoaringBitmap::from_iter([1u32, 2]));
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;
}

/// Seed records are never larger than the default zone size, so a seed
/// written for a large-zone index still serves a smaller one later.
#[tokio::test]
async fn test_seed_records_are_capped_at_the_default_zone_size() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(1_000_000, true, true),
        None,
    )
    .await;
    append(&mut dataset, N..2 * N).await;
    assert_eq!(
        seed_locator(&dataset, 1, "val")
            .await
            .as_deref()
            .unwrap()
            .split(':')
            .nth(1),
        Some("8192")
    );
    create_index(
        &mut dataset,
        "name",
        &zone_map_params(100, true, true),
        None,
    )
    .await;
    append(&mut dataset, 2 * N..3 * N).await;
    assert_eq!(
        seed_locator(&dataset, 2, "name")
            .await
            .as_deref()
            .unwrap()
            .split(':')
            .nth(1),
        Some("100")
    );

    // 8192-row records do not fit 1000-row zones, so a 1000-row index scans.
    dataset.drop_index(&index_name("val")).await.unwrap();
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(1000, true, true),
        None,
    ))
    .await;
    assert!(reports[0].reused_fragments.is_empty());
    assert_eq!(reports[0].scanned_fragments, vec![0, 1, 2]);
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;
}

/// Floating-point seeds with NaN of both signs and infinities fold into
/// large zones and keep index results equal to a scan. This test uses only
/// public APIs so the same assertions run on the base revision.
#[tokio::test]
async fn test_float_nan_queries_match_scan_after_seed_reuse() {
    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(&mut dataset, "x", &zone_map_params(8, true, true), None).await;
    append(&mut dataset, N..2 * N).await;
    append(&mut dataset, 2 * N..3 * N).await;
    dataset.drop_index(&index_name("x")).await.unwrap();
    create_index(&mut dataset, "x", &zone_map_params(20000, true, true), None).await;
    assert_index_matches_scan(
        &dataset,
        &[
            "x < 0",
            "x > 0",
            "x >= 1.5",
            "x <= -40",
            "x = 2",
            "x = 48",
            "x BETWEEN -1 AND 1",
            "x IN (3, 5, 1e308)",
            "x IS NULL",
            "x IS NOT NULL",
            "x > 1e308",
            "x < -1e308",
        ],
    )
    .await;
}

/// A current overlay on the indexed column changes values the base file's
/// seed never saw, so that fragment is scanned, at creation and on update.
/// An overlay on another column leaves the seed usable.
#[tokio::test]
async fn test_overlaid_fragment_is_scanned_instead_of_reused() {
    use super::dataset_overlay_index_masking::commit_overlay;
    use lance_table::format::overlay::OverlayCoverage;

    let dir = TempStrDir::default();
    let mut dataset = write(dir.as_str(), 0..N).await;
    create_index(&mut dataset, "val", &zone_map_params(8, true, true), None).await;
    append(&mut dataset, N..2 * N).await;
    append(&mut dataset, 2 * N..3 * N).await;
    let val_field = dataset.schema().field("val").unwrap().id;
    let name_field = dataset.schema().field("name").unwrap().id;
    // Fragment 1, offset 0 (id = N): val becomes 10,000,000, beyond every seed bound.
    let dataset = commit_overlay(
        dataset,
        "val_overlay",
        1,
        &[val_field],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0])),
        vec![Arc::new(Int64Array::from(vec![Some(10_000_000)]))],
    )
    .await;
    // Fragment 2 gets an overlay on an unrelated column.
    let mut dataset = commit_overlay(
        dataset,
        "name_overlay",
        2,
        &[name_field],
        OverlayCoverage::dense(RoaringBitmap::from_iter([0])),
        vec![Arc::new(StringArray::from(vec![Some("overlaid")]))],
    )
    .await;

    // Creation: fragment 1 is scanned, fragment 2 keeps using its seed.
    dataset.drop_index(&index_name("val")).await.unwrap();
    let (_, reports) = with_reports(create_index(
        &mut dataset,
        "val",
        &zone_map_params(N as u64, true, true),
        None,
    ))
    .await;
    assert_eq!(reports[0].reused_fragments, vec![2]);
    assert_eq!(reports[0].scanned_fragments, vec![0, 1]);
    assert_eq!(
        ids_matching(&dataset, "val = 10000000", true, true).await,
        BTreeSet::from([N])
    );
    assert_eq!(
        ids_matching(&dataset, "val > 9000000", true, true).await,
        BTreeSet::from([N])
    );
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;

    // Update: the index covers fragment 0 only; maintenance scans fragment 1
    // and reuses fragment 2.
    create_index(
        &mut dataset,
        "val",
        &zone_map_params(N as u64, true, true),
        Some(vec![0]),
    )
    .await;
    let (_, reports) = with_reports(async {
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
    })
    .await;
    assert_eq!(reports[0].reused_fragments, vec![2]);
    assert_eq!(reports[0].scanned_fragments, vec![1]);
    let segments = index_metadata(&dataset, "val").await;
    assert_eq!(coverage(&segments[0]), RoaringBitmap::from_iter(0..3u32));
    assert_eq!(
        ids_matching(&dataset, "val = 10000000", true, true).await,
        BTreeSet::from([N])
    );
    assert_index_matches_scan(&dataset, VAL_FILTERS).await;
}
