# Updated notebook cells 8-13 to use the new EnsembleRunner API

# Cell 8: Load data and build initial partition (REPLACE entire cell)
from utgc.ensemble import EnsembleRunner

# Create and run ensemble analysis
runner = EnsembleRunner(config)
results = runner.run(output_dir=config_dir)

print(f"Ensemble analysis complete! Generated {len(results)} maps.")
print(f"Results saved to: {config_dir}")

# Cell 9: Create constraints and proposal (REMOVE this cell entirely)
# Constraints and proposal are now handled in EnsembleRunner

# Cell 10: Preconditioning step (REMOVE this cell entirely) 
# Preconditioning is now handled in EnsembleRunner

# Cell 11: Run ensemble sample and visualize (REMOVE this cell entirely)
# Ensemble running is now handled in EnsembleRunner

# Cell 12: Inline plots of partition-wide metrics (UPDATE to use correct parameter names)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rows = []
for r in results:
    step = r.get("step")
    split_m = r.get("split_munis_count")
    split_c = r.get("split_counties_count")
    split_m_multi = r.get("split_munis_extra_parts", 0)
    split_c_multi = r.get("split_counties_extra_parts", 0)
    cut_e = r.get("num_cut_edges")
    pop_dev = r.get("population_deviation", {})
    try:
        dev_vals = [v for v in pop_dev.values() if v is not None]
        max_dev = max(dev_vals) if dev_vals else None
        mean_dev = sum(dev_vals) / len(dev_vals) if dev_vals else None
    except Exception:
        max_dev = None
        mean_dev = None
    rows.append({
        "step": step,
        "split_munis_count": split_m,
        "split_counties_count": split_c,
        "split_munis_multi": split_m_multi,
        "split_counties_multi": split_c_multi,
        "population_max_dev": max_dev,
        "population_mean_dev": mean_dev,
        "num_cut_edges": cut_e,
    })

metrics_df = pd.DataFrame(rows).sort_values("step")
print("DataFrame columns:", list(metrics_df.columns))
print("DataFrame shape:", metrics_df.shape)
print("Sample data:")
print(metrics_df.head())

sns.set_style("whitegrid")
fig, axes = plt.subplots(4, 1, sharex=True)

axes[0].plot(metrics_df["step"], metrics_df["population_max_dev"], label="Max pop dev", color="#1f77b4")
axes[0].plot(metrics_df["step"], metrics_df["population_mean_dev"], label="Mean pop dev", color="#ff7f0e")
axes[0].set_ylabel("Pop. dev.")

# Plot municipality splits
if "split_munis_count" in metrics_df.columns:
    axes[1].plot(metrics_df["step"], metrics_df["split_munis_count"], label="Muni splits", color="#2ca02c")
if "split_munis_multi" in metrics_df.columns:
    axes[1].plot(metrics_df["step"], metrics_df["split_munis_multi"], label="Muni multi-splits", color="#2ca02c", linestyle="--")
axes[1].set_ylabel("Muni splits")
axes[1].legend()

# Plot county splits
if "split_counties_count" in metrics_df.columns:
    axes[2].plot(metrics_df["step"], metrics_df["split_counties_count"], label="County splits", color="#d62728")
if "split_counties_multi" in metrics_df.columns:
    axes[2].plot(metrics_df["step"], metrics_df["split_counties_multi"], label="County multi-splits", color="#d62728", linestyle="--")
axes[2].set_ylabel("County splits")
axes[2].legend()

axes[3].plot(metrics_df["step"], metrics_df["num_cut_edges"], label="Cut edges", color="#6b7280")
axes[3].set_ylabel("Cut edges")
axes[3].set_xlabel("Step")

fig.set_dpi(300)
plt.tight_layout()
plt.show()

# Cell 13: Interactive visualization (UPDATE to use correct parameter names)
import os
from IPython.display import display
import ipywidgets as widgets
from PIL import Image

# Load all images saved in the config_dir directory (assumed to be PNGs)
image_dir = str(config_dir)
image_files = sorted(
    [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
)

img = widgets.Image(format='png')

steps = list(range(0, ensemble_params["steps"], ensemble_params["visualize_every"]))
steps2index = {s: i for i, s in enumerate(steps)}
# Slider shows actual step numbers
stepper = widgets.BoundedIntText(value=steps[0], min=min(steps), max=max(steps), step=ensemble_params["visualize_every"], description="Step:")

frames = []
for fname in image_files:
    with open(os.path.join(image_dir, fname), "rb") as f:
        data = f.read()
    frames.append(data)

# Ensure no duplicate observers if you re-run this cell
try:
    stepper.unobserve_all()
except Exception: pass

def on_change(value):
    img.value = frames[steps2index[value]]

widgets.interactive(on_change, value=stepper)
img.value = frames[steps2index[stepper.value]]

# Create and display the widget
widget_box = widgets.VBox([stepper, img])
display(widget_box)

# Cell 14: Export YAML configuration (KEEP this cell as-is - it's already correct)

