# UT_GerryChain YAML-first configuration (transitional)

Ensemble runs now prefer YAML configuration. CLI flags still work for a short transition period, but running without YAML will print a deprecation warning and will be removed in a future release.

## Configure neutral sampling in notebook

Use `03_configure_sampling.ipynb` to:
- Configure neutral parameters (no partisan metrics)
- Configure transitability analysis (road connectivity and water barriers)
- Run a short neutral sample for sanity checks
- Export a YAML configuration to `results/configurations/sampling_params.yaml`

## Transitability Analysis

UT_GerryChain now includes transitability analysis to ensure districts are connected by actual roads and not separated by impassable barriers. This implements Utah's requirement for "ease of travel throughout district" by:

1. **Road Connectivity**: Verifying that precincts are connected by actual road networks
2. **Hierarchical Fallback**: For rural areas with only local roads, fall back to municipality/county boundaries  
3. **Water Barriers**: Remove connections that cross major water bodies (Great Salt Lake, Lake Powell, etc.)

### Data Requirements

You can now build the transitability graph offline from the original geography sources and load it at runtime.

### Offline preprocessing (recommended)

Build once and reuse:

```bash
# 1) Build artifacts under data/transitability
python data/scripts/transitability/build_transitability_graph.py \
  --precincts data/UT_precincts.geojson \
  --lakes data/geography/UtahMajorLakes/UtahMajorLakes.shp \
  --rivers data/geography/UtahMajorRiversPoly/UtahMajorRivers.shp \
  --roads data/geography/UtahRoads/UtahRoads.shp \
  --water-threshold 0.05 \
  --water-buffer-m 150 \
  --road-boundary-buffer-m 15

# 2) (Optional) Visual overview figure
python data/scripts/transitability/viz_transitability.py --save
```

Artifacts written to `data/transitability/`:
- `edges_base.parquet`, `edges_final.parquet`
- `transitability_graph_edges.parquet`, `transitability_graph_nodes.parquet`
- `transitability.graphml`, `transitability.json`
- `metadata.yaml`

### Configuration

Transitability can be configured in the notebook or YAML:

```yaml
transitability:
  enable: true
  remove_water_barriers: true
  verify_road_connectivity: true
  precomputed_path: data/transitability/transitability.graphml
  min_lake_size_sqkm: 1.0
  min_river_size_sqkm: 0.5
  road_buffer_meters: 500
  water_threshold: 0.5
```

## Run with YAML (preferred)

```bash
python 04_run_ensemble.py --config results/configurations/sampling_params.yaml
```

Flags can still override YAML during the transition:
```bash
python 04_run_ensemble.py --config results/configurations/sampling_params.yaml --steps 200 --viz-every 5
```

Precedence: CLI flags > YAML values > code defaults.

When `precomputed_path` is provided, the runtime will load the prebuilt graph (GraphML or JSON edge list).

## Example YAML

```yaml
steps: 200
viz_every: 10
use_cut_edges: true
max_muni_splits: 3
max_county_splits: 2
muni_surcharge: 9
county_surcharge: 3
highered_surcharge: 1
metro_surcharge: 1
schdist_surcharge: 0.1
water_surcharge: 0.1
basin_surcharge: 0.1
tilted_run: 0.0
# Optional (used by CLI if present)
years: "2016,2020,2024"
offices: "PRE,GOV,ATG,AUD,TRE"
vote_share_agg: "median"
```

## Notes
- When no elections are provided (empty after filtering), outputs are saved in neutral mode (no partisan statistics in the summary CSV).
- Future release will require YAML-only configuration.
