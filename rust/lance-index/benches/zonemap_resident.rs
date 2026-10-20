// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Summary-scale measurements without generating or retaining the underlying rows.
//! Prepare a fixture once, then measure each revision in a fresh process against
//! that same directory. Use an external process monitor for load peak RSS.
//!
//! ZONEMAP_DIRECTORY=/tmp/zonemap-i64 ZONEMAP_PREPARE=1 \
//!   cargo bench -p lance-index --bench zonemap_resident --profile release-with-debug
//! ZONEMAP_DIRECTORY=/tmp/zonemap-i64 \
//!   cargo bench -p lance-index --bench zonemap_resident --profile release-with-debug
//! Optional: ZONEMAP_SUMMARIES (default 1000000), ZONEMAP_TYPE (int32, int64, timestamp,
//! utf8), ZONEMAP_NULL_EVERY (default 0; otherwise every Nth zone has one null).
//! ZONEMAP_LOAD_ONLY=1 exits immediately after loading, before query result masks
//! can contribute to the externally measured peak RSS.
//! ZONEMAP_UPDATE_DIRECTORY writes an empty-data update there and measures the
//! maintenance peak separately from queries.
//! ZONEMAP_QUERY selects one query (miss, selective, null, broad).

#![allow(clippy::print_stdout)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_common::ScalarValue;
use lance_core::cache::LanceCache;
use lance_core::deepsize::DeepSizeOf;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::pbold;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::ScalarIndexPlugin;
use lance_index::scalar::zonemap::ZoneMapIndexPlugin;
use lance_index::scalar::{IndexStore, SargableQuery};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;

const ROWS_PER_ZONE: u64 = 1_000_000;

struct TrackingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

fn allocated(bytes: usize) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: Every allocation is forwarded to System unchanged. The counters never
// inspect allocated memory and are independent of allocation alignment/lifetime.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() {
            if new_size >= layout.size() {
                allocated(new_size - layout.size());
            } else {
                LIVE_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        pointer
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[derive(Clone, Copy, Debug)]
enum ValueType {
    Int32,
    Int64,
    Timestamp,
    Utf8,
}

#[derive(Debug)]
struct Options {
    directory: PathBuf,
    prepare: bool,
    load_only: bool,
    update_directory: Option<PathBuf>,
    query: Option<String>,
    summaries: usize,
    value_type: ValueType,
    null_every: usize,
}

impl Options {
    fn from_env() -> Self {
        Self {
            directory: std::env::var_os("ZONEMAP_DIRECTORY")
                .expect("set ZONEMAP_DIRECTORY")
                .into(),
            prepare: std::env::var_os("ZONEMAP_PREPARE").is_some(),
            load_only: std::env::var_os("ZONEMAP_LOAD_ONLY").is_some(),
            update_directory: std::env::var_os("ZONEMAP_UPDATE_DIRECTORY").map(Into::into),
            query: std::env::var("ZONEMAP_QUERY").ok(),
            summaries: std::env::var("ZONEMAP_SUMMARIES")
                .map(|v| v.parse().unwrap())
                .unwrap_or(1_000_000),
            value_type: match std::env::var("ZONEMAP_TYPE").as_deref().unwrap_or("int64") {
                "int32" => ValueType::Int32,
                "int64" => ValueType::Int64,
                "timestamp" => ValueType::Timestamp,
                "utf8" => ValueType::Utf8,
                value => panic!("invalid ZONEMAP_TYPE: {value}"),
            },
            null_every: std::env::var("ZONEMAP_NULL_EVERY")
                .map(|v| v.parse().unwrap())
                .unwrap_or(0),
        }
    }
}

fn string_value(value: i64) -> String {
    format!("key-{value:020}-x")
}

fn scalar(value_type: ValueType, value: i64) -> ScalarValue {
    match value_type {
        ValueType::Int32 => ScalarValue::Int32(Some(value.try_into().unwrap())),
        ValueType::Int64 => ScalarValue::Int64(Some(value)),
        ValueType::Timestamp => ScalarValue::TimestampMicrosecond(Some(value), Some("UTC".into())),
        ValueType::Utf8 => ScalarValue::Utf8(Some(string_value(value))),
    }
}

async fn prepare(options: &Options, store: &dyn IndexStore) {
    let numeric = Int64Array::from(
        (0..options.summaries)
            .map(|i| 2 * i as i64)
            .collect::<Vec<_>>(),
    );
    let (min, max): (ArrayRef, ArrayRef) = match options.value_type {
        ValueType::Int32 => (
            Arc::new(arrow_array::Int32Array::from(
                (0..options.summaries)
                    .map(|i| (2 * i).try_into().unwrap())
                    .collect::<Vec<i32>>(),
            )),
            Arc::new(arrow_array::Int32Array::from(
                (0..options.summaries)
                    .map(|i| (2 * i + 1).try_into().unwrap())
                    .collect::<Vec<i32>>(),
            )),
        ),
        ValueType::Int64 => (
            Arc::new(numeric),
            Arc::new(Int64Array::from(
                (0..options.summaries)
                    .map(|i| 2 * i as i64 + 1)
                    .collect::<Vec<_>>(),
            )),
        ),
        ValueType::Timestamp => (
            Arc::new(
                numeric
                    .reinterpret_cast::<arrow_array::types::TimestampMicrosecondType>()
                    .with_timezone("UTC"),
            ),
            Arc::new(
                arrow_array::TimestampMicrosecondArray::from(
                    (0..options.summaries)
                        .map(|i| 2 * i as i64 + 1)
                        .collect::<Vec<_>>(),
                )
                .with_timezone("UTC"),
            ),
        ),
        ValueType::Utf8 => (
            Arc::new(StringArray::from_iter_values(
                (0..options.summaries).map(|i| string_value(2 * i as i64)),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..options.summaries).map(|i| string_value(2 * i as i64 + 1)),
            )),
        ),
    };
    let data_type = min.data_type().clone();
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("min", data_type.clone(), true),
            Field::new("max", data_type, true),
            Field::new("null_count", DataType::UInt32, false),
            Field::new("nan_count", DataType::UInt32, false),
            Field::new("fragment_id", DataType::UInt64, false),
            Field::new("zone_start", DataType::UInt64, false),
            Field::new("zone_length", DataType::UInt64, false),
        ],
        HashMap::from([("rows_per_zone".into(), ROWS_PER_ZONE.to_string())]),
    ));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            min,
            max,
            Arc::new(UInt32Array::from(
                (0..options.summaries)
                    .map(|i| u32::from(options.null_every != 0 && i % options.null_every == 0))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(vec![0; options.summaries])),
            Arc::new(UInt64Array::from(
                (0..options.summaries as u64).collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(vec![0; options.summaries])),
            Arc::new(UInt64Array::from(vec![ROWS_PER_ZONE; options.summaries])),
        ],
    )
    .unwrap();
    let mut file = store.new_index_file("zonemap.lance", schema).await.unwrap();
    file.write_record_batch(batch).await.unwrap();
    file.finish_with_metadata(HashMap::new()).await.unwrap();
}

fn rss_bytes() -> Option<u64> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}

#[tokio::main]
async fn main() {
    let options = Options::from_env();
    if options.prepare {
        std::fs::create_dir_all(&options.directory).unwrap();
    }
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        Path::from_filesystem_path(&options.directory).unwrap(),
        Arc::new(LanceCache::no_cache()),
    ));
    if options.prepare {
        prepare(&options, store.as_ref()).await;
        return;
    }
    let rss_before = rss_bytes();
    let allocated_before = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(allocated_before, Ordering::Relaxed);
    let started = Instant::now();
    let details = prost_types::Any::from_msg(&pbold::ZoneMapIndexDetails::default()).unwrap();
    let index = ZoneMapIndexPlugin
        .load_index(store, &details, 0, None, &LanceCache::no_cache())
        .await
        .unwrap();
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let allocated_loaded = LIVE_BYTES.load(Ordering::Relaxed);
    let allocated_load_peak = PEAK_BYTES.load(Ordering::Relaxed);
    let rss_loaded = rss_bytes();
    let mut report = serde_json::json!({
        "value_type": format!("{:?}", options.value_type), "summaries": options.summaries,
        "null_every": options.null_every, "resident_bytes": index.deep_size_of(),
        "rss_before_bytes": rss_before, "rss_loaded_bytes": rss_loaded,
        "load_ms": load_ms,
        "allocated_before_bytes": allocated_before,
        "allocated_loaded_bytes": allocated_loaded,
        "allocated_load_peak_bytes": allocated_load_peak,
    });
    if options.load_only {
        println!("{report}");
        return;
    }
    if let Some(directory) = &options.update_directory {
        std::fs::create_dir_all(directory).unwrap();
        let destination = LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(directory).unwrap(),
            Arc::new(LanceCache::no_cache()),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", scalar(options.value_type, 0).data_type(), true),
            Field::new(lance_core::ROW_ADDR, DataType::UInt64, false),
        ]));
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::empty(),
        ));
        let allocated_start = LIVE_BYTES.load(Ordering::Relaxed);
        PEAK_BYTES.store(allocated_start, Ordering::Relaxed);
        let started = Instant::now();
        black_box(index.update(data, &destination, None).await.unwrap());
        let update_ms = started.elapsed().as_secs_f64() * 1000.0;
        report["update_ms"] = serde_json::json!(update_ms);
        report["allocated_update_start_bytes"] = serde_json::json!(allocated_start);
        report["allocated_update_peak_bytes"] =
            serde_json::json!(PEAK_BYTES.load(Ordering::Relaxed));
        report["rss_updated_bytes"] = serde_json::json!(rss_bytes());
        println!("{report}");
        return;
    }
    let queries = [
        (
            "miss",
            SargableQuery::Equals(scalar(options.value_type, -2)),
        ),
        (
            "selective",
            SargableQuery::Equals(scalar(options.value_type, options.summaries as i64)),
        ),
        ("null", SargableQuery::IsNull()),
        (
            "broad",
            SargableQuery::Range(std::ops::Bound::Unbounded, std::ops::Bound::Unbounded),
        ),
    ];
    let mut measurements = Vec::new();
    for (name, query) in queries.into_iter().filter(|(name, _)| {
        options
            .query
            .as_deref()
            .is_none_or(|chosen| chosen == *name)
    }) {
        let mut samples = Vec::with_capacity(11);
        let mut result_size = 0;
        for _ in 0..11 {
            let started = Instant::now();
            let result = black_box(
                index
                    .search(black_box(&query), &NoOpMetricsCollector)
                    .await
                    .unwrap(),
            );
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            if result_size == 0 {
                result_size = result.row_addrs().selected_rows().deep_size_of();
            }
        }
        samples.sort_by(f64::total_cmp);
        measurements.push(serde_json::json!({"query": name, "median_ms": samples[5], "min_ms": samples[0], "max_ms": samples[10], "result_bytes": result_size}));
    }
    let started = Instant::now();
    black_box(index.value_range());
    let value_range_ms = started.elapsed().as_secs_f64() * 1000.0;
    report["queries"] = serde_json::json!(measurements);
    report["value_range_ms"] = serde_json::json!(value_range_ms);
    println!("{report}");
}
