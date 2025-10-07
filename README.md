# UT_GerryChain YAML-first configuration (transitional)

Ensemble runs now prefer YAML configuration. CLI flags still work for a short transition period, but running without YAML will print a deprecation warning and will be removed in a future release.

## Configure neutral sampling in notebook

Use `03_configure_sampling.ipynb` to:
- Configure neutral parameters (no partisan metrics)
- Run a short neutral sample for sanity checks
- Export a YAML configuration to `results/configurations/sampling_params.yaml`

## Run with YAML (preferred)

```bash
python 04_run_ensemble.py --config results/configurations/sampling_params.yaml
```

Flags can still override YAML during the transition:
```bash
python 04_run_ensemble.py --config results/configurations/sampling_params.yaml --steps 200 --viz-every 5
```

Precedence: CLI flags > YAML values > code defaults.

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
