// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::sync::Arc;

use arrow_schema::DataType;
use itertools::Itertools;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::zonemap::{
    HarvestedZoneSeed, ReusedZones, ZoneMapIndex, ZoneMapIndexBuilderParams, ZoneMapSegmentBase,
    build_zonemap_segment,
};
use lance_index::scalar::{
    CreatedIndex, IndexStore, ScalarIndex, TrainingCriteria, index_files_to_table,
};
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::index::append::SeedHarvester;
use crate::index::scalar::load_training_data;
use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Merge one caller-defined group of source ZoneMap segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    if segments.is_empty() {
        return Err(Error::index("No segment metadata was provided".to_string()));
    }

    let field_id = *segments[0].fields.first().ok_or_else(|| {
        Error::invalid_input(format!(
            "CreateIndex: segment {} is missing field ids",
            segments[0].uuid
        ))
    })?;
    let field_path = dataset.schema().field_path(field_id)?;

    let mut scalar_indices = Vec::with_capacity(segments.len());
    let mut fragment_bitmap = RoaringBitmap::new();
    let dataset_fragments = dataset.fragment_bitmap.as_ref();
    for segment in &segments {
        let effective = segment
            .effective_fragment_bitmap(dataset_fragments)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "CreateIndex: segment {} is missing fragment coverage",
                    segment.uuid
                ))
            })?;
        fragment_bitmap |= effective;
        let scalar_index = super::open_scalar_index_with_plan(
            dataset,
            &field_path,
            segment,
            staged.and_then(|plans| plans.get(&segment.uuid)),
            crate::index::frag_reuse::OpenPurpose::Maintenance,
            &NoOpMetricsCollector,
        )
        .await?;
        scalar_indices.push((segment.uuid, scalar_index));
    }

    let mut source_indices = Vec::with_capacity(scalar_indices.len());
    for (segment_uuid, scalar_index) in &scalar_indices {
        let zonemap_index = scalar_index
            .as_any()
            .downcast_ref::<ZoneMapIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "merge_existing_index_segments: expected zonemap segment {}, got {:?}",
                    segment_uuid,
                    scalar_index.index_type()
                ))
            })?;
        source_indices.push(zonemap_index);
    }

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    let created_index = lance_index::scalar::zonemap::merge_zonemap_indices(
        &source_indices,
        &new_store,
        &fragment_bitmap,
    )
    .await?;

    Ok(IndexMetadata {
        uuid: new_uuid,
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(created_index.index_details)),
        index_version: created_index.index_version as i32,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: Some(index_files_to_table(created_index.files)),
        ..segments[0].clone()
    })
}

/// What a seed-aware segment build reused and scanned, recorded for tests.
#[cfg(test)]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeedReuseReport {
    pub reused_fragments: Vec<u64>,
    pub reused_records: usize,
    pub zones_from_seeds: usize,
    pub scanned_fragments: Vec<u64>,
    /// Largest raw seed buffer held at any moment during the build.
    pub max_live_seed_bytes: usize,
    /// Raw seed bytes harvested over the whole build.
    pub total_seed_bytes: usize,
}

#[cfg(test)]
tokio::task_local! {
    /// Collects one [`SeedReuseReport`] per seed-aware segment build.
    pub static SEED_REUSE_REPORTS: std::sync::Mutex<Vec<SeedReuseReport>>;
}

/// Fold the usable seeds into zones, scan the remaining fragments, and write
/// one segment.
///
/// Seeds are harvested, validated and folded one fragment at a time, so the
/// raw seed buffer held in memory is at most one fragment's, whatever the
/// table size; only the packed zones (and, when kept, null positions)
/// accumulate.
#[allow(clippy::too_many_arguments)]
async fn assemble_segment(
    dataset: &Dataset,
    column: &str,
    base: ZoneMapSegmentBase<'_>,
    data_type: &DataType,
    fragments: &[Fragment],
    harvester: Option<SeedHarvester<'_>>,
    criteria: &TrainingCriteria,
    store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let mut reused = ReusedZones::new(base.keeps_null_positions());
    let (mut max_live_seed_bytes, mut total_seed_bytes) = (0usize, 0usize);
    if let Some(mut harvester) = harvester {
        for fragment in fragments {
            // Without a known physical row count the seed's coverage cannot
            // be verified; the fragment is scanned.
            let Some(physical_rows) = fragment.physical_rows else {
                continue;
            };
            let Some(seed) = harvester.harvest(fragment).await? else {
                continue;
            };
            let harvested = HarvestedZoneSeed {
                seed,
                physical_rows: physical_rows as u64,
            };
            let bytes = harvested.seed.bytes.len();
            max_live_seed_bytes = max_live_seed_bytes.max(bytes);
            total_seed_bytes += bytes;
            reused.reuse(&harvested, base.rows_per_zone(), data_type)?;
            // `harvested` is dropped here: the raw buffer does not outlive
            // its fragment's fold.
        }
    }
    let reused_ids: HashSet<u64> = reused.reused_fragments.iter().copied().collect();
    let to_scan: Vec<Fragment> = fragments
        .iter()
        .filter(|fragment| !reused_ids.contains(&fragment.id))
        .cloned()
        .collect();
    log::info!(
        "zone map segment for {column}: {} fragments from {} seed records ({total_seed_bytes} seed bytes, at most {max_live_seed_bytes} held at once), {} fragments scanned",
        reused.reused_fragments.len(),
        reused.reused_records,
        to_scan.len()
    );
    #[cfg(test)]
    let _ = SEED_REUSE_REPORTS.try_with(|reports| {
        reports.lock().unwrap().push(SeedReuseReport {
            reused_fragments: reused.reused_fragments.clone(),
            reused_records: reused.reused_records,
            zones_from_seeds: reused.num_zones(),
            scanned_fragments: to_scan.iter().map(|fragment| fragment.id).collect(),
            max_live_seed_bytes,
            total_seed_bytes,
        })
    });
    let scan = if to_scan.is_empty() {
        None
    } else {
        Some(load_training_data(dataset, column, criteria, Some(to_scan), true, None).await?)
    };
    let refreshed: Vec<u64> = fragments.iter().map(|fragment| fragment.id).collect();
    build_zonemap_segment(base, data_type, reused, scan, &refreshed, store).await
}

/// Train a new ZoneMap segment over `fragment_ids` (all fragments when
/// `None`), reusing the write seeds those fragments already carry and
/// scanning only the rest.
pub(in crate::index) async fn build_segment_with_seeds(
    dataset: &Dataset,
    column: &str,
    params: ZoneMapIndexBuilderParams,
    data_type: &DataType,
    fragment_ids: Option<Vec<u32>>,
    criteria: &TrainingCriteria,
    store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let fragments: Vec<Fragment> = match fragment_ids {
        Some(ids) => {
            let ids: Vec<u32> = ids.into_iter().sorted().dedup().collect();
            ids.iter()
                .zip(dataset.get_frags_from_ordered_ids(&ids))
                .map(|(id, fragment)| {
                    fragment.map(|f| f.metadata().clone()).ok_or_else(|| {
                        Error::invalid_input_source(format!("No fragment with id {}", id).into())
                    })
                })
                .collect::<Result<_>>()?
        }
        None => dataset.fragments().as_ref().clone(),
    };
    assemble_segment(
        dataset,
        column,
        ZoneMapSegmentBase::New(params),
        data_type,
        &fragments,
        SeedHarvester::for_column(dataset, column),
        criteria,
        store,
    )
    .await
}

/// Append `unindexed` fragments to the ZoneMap segment `reference_index`,
/// reusing their write seeds where usable. `None` when the reference is not
/// a ZoneMap segment or its index is not keyed on a top-level field, in which
/// case the generic update path applies.
#[allow(clippy::too_many_arguments)]
pub(in crate::index) async fn update_segment_with_seeds(
    dataset: &Dataset,
    reference_index: &Arc<dyn ScalarIndex>,
    reference: &IndexMetadata,
    column: &str,
    unindexed: &[Fragment],
    criteria: &TrainingCriteria,
    store: &dyn IndexStore,
) -> Result<Option<CreatedIndex>> {
    let Some(zone_map) = reference_index.as_any().downcast_ref::<ZoneMapIndex>() else {
        return Ok(None);
    };
    let Some(harvester) = reference
        .keyed_field()
        .and_then(|field_id| SeedHarvester::for_field_id(dataset, field_id))
    else {
        return Ok(None);
    };
    let data_type = zone_map.value_type().clone();
    assemble_segment(
        dataset,
        column,
        ZoneMapSegmentBase::Update(zone_map),
        &data_type,
        unindexed,
        Some(harvester),
        criteria,
        store,
    )
    .await
    .map(Some)
}
