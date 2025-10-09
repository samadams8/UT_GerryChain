<!-- 5949fa8a-6b43-4947-8581-0db1f556f001 5d5db2ce-9179-4bc2-9f92-1313ab96a553 -->
# Decisions confirmed

- Export formats: GraphML for maximum interoperability with NetworkX/GerryChain, plus Parquet (edges/nodes) and JSON edge list.
- Water threshold: any intersection > 0 for major waters; 0.05 otherwise.

### Scope

- Build an offline preprocessing pipeline that ingests precincts, roads, lakes/rivers; computes a transitability-aware graph; generates step-by-step visuals; and writes reusable artifacts.
- Refactor: move transitability logic out of runtime path into `data/scripts/transitability/` and provide a minimal loader in `utgc/build.py` to import a precomputed graph when requested.

### Data inputs

- Precincts: `data/UT_precincts.geojson`
- Roads (filtered): `data/geography_processed/UtahRoads_filtered.*`
- Water (filtered): `data/geography_processed/UtahMajorLakes_filtered.*`, `UtahMajorRivers_filtered.*`

### Outputs

- Precomputed graph artifacts in `results/transitability/`:
- `transitability_graph_edges.parquet` (u,v, attributes)
- `transitability_graph_nodes.parquet` (node_id, attributes)
- `transitability.graphml`
- `transitability.json` (edge list with attributes)
- Per-step edge lists: `edges_base.parquet`, `edges_road.parquet`, `edges_water.parquet`, `edges_final.parquet`
- Visuals in `results/transitability/plots/`:
- Base adjacency, road-pruned, water-pruned, final + deltas (removed-by-road, removed-by-water)
- Also save each panel individually after each step
- County-focused panels: Kane–San Juan, Great Salt Lake, Colorado River/Lake Powell corridors
- Provenance: `results/transitability/metadata.yaml` (CRS, thresholds, buffers, dataset versions)

### Algorithm (consolidated best-of-learnings)

1) Load and normalize

- Read precincts; validate CRS; index on precinct id
- Read roads/water; ensure CRS matches precincts; fix invalid geometries; optional snap/repair (`maup.smart_repair` with low aggression)

2) Base adjacency

- Compute base adjacency using `maup.adjacencies(precincts, adjacency_type='rook')`
- Keep as candidate edge list with shared-boundary geometry (store boundary length)

3) Water pruning (shared-boundary first)

- Buffer water polygons by 100–200 m; union to fill slivers
- For each candidate shared boundary (from maup.adjacencies), compute intersection length with buffered water
- Remove edge if intersection_ratio ≥ water_threshold (0.05 default)
- For major waters (Lake Powell|Colorado River|Great Salt Lake), remove if any intersection > 0

4) Road verification and add-backs (pairwise)

- Build a light road network graph from linework (nodes at segment endpoints; edges with length/type)
- For each pruned edge, check if there exists a road that actually bridges the shared boundary:
- Fast heuristic: any road segment intersects the shared boundary and touches precinct A and B buffers (e.g., 50 m)
- Robust: attempt a shortest path from a point on A near the boundary to a point on B near the boundary constrained to a corridor (e.g., 300 m) across the boundary; if found, ADD BACK the edge with attribute `added_by_road=True`
- Also allow manual add-back list from a CSV of (u,v)

5) Hierarchical fallback (last step)

- For precincts now isolated, connect to same-county neighbors along base adjacency, but do not violate major-water rule unless road add-back confirmed

6) Validate and export

- Confirm final graph connectivity; log components if not fully connected
- Export nodes/edges; write plots and metadata

### Visuals to generate

- 6-panel overview (Base, Road-pruned, Water-pruned, Final, Removed-by-road, Removed-by-water)
- Annotated insets for: Great Salt Lake; Lake Powell/Colorado; Kane–San Juan interface; other long-bridge candidates
- CSV summaries: counts and percentages removed per county and per major-water zone

### Repository refactor

- New directory: `data/scripts/transitability/`
- `build_transitability_graph.py` (main entrypoint)
- `viz_transitability.py` (visuals)
- `utils.py` (adjacency, cross-boundary tests, IO)
- Minimal loader update (optional): `utgc/build.py` can load precomputed graph when `config['transitability']['precomputed_path']` is set; otherwise falls back to standard graph
- Deprecate on-the-fly transitability path in ensemble

### Configurability

- CLI flags and YAML:
- `--water-threshold` (float)
- `--water-buffer-m` (int, default 150)
- `--road-boundary-buffer-m` (int, default 15)
- `--major-waters` list of regexes
- `--export-formats` [parquet, graphml, json]
- `--areas` optional county focus list for extra visuals

### Notes

- Performance: vectorize per-step; avoid per-edge loops where possible; spatial index joins for boundary-road and boundary-water checks
- Determinism: write seed, CRS, and file mtimes into metadata

### To-dos

- [ ] Create 05_prepare_transitability_data.py to filter and process geographic datasets
- [x] Run data prep script to generate filtered water bodies and roads datasets
- [x] Create utgc/transitability.py with Phase 1 water barrier detection functions
- [x] Implement water barrier edge removal logic
- [x] Implement road network connectivity verification
- [x] Modify utgc/build.py to integrate transitability into graph creation
- [x] Add transitability configuration section to 03_configure_sampling.ipynb
- [x] Test transitability with sample runs and validate edge removal accuracy
- [x] Update README with transitability feature documentation and dataset instructions