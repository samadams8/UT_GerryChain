import os
import yaml
import geopandas as gpd
import pandas as pd
from IPython.display import display
import ipywidgets as widgets
from PIL import Image

def get_district_count(shapefile_path):
    """
    Get the number of districts from a shapefile.
    """
    shapefile = gpd.read_file(shapefile_path)
    return len(shapefile)

def load_config(config_path=""):
    """
    Load a configuration file. If none is provided, we'll retrieve the latest in results/configurations/
    """
    if config_path == "":
        configs_dir = os.path.join("results", "configurations")
        if not os.path.isdir(configs_dir):
            return None
        candidates = []
        for root, _, files in os.walk(configs_dir):
            for name in files:
                if name.endswith(".yaml") or name.endswith(".yml"):
                    path = os.path.join(root, name)
                    try:
                        mtime = os.path.getmtime(path)
                    except Exception:
                        mtime = 0
                    candidates.append((mtime, path))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        config_path = candidates[0][1]
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def map_viewer_widget(image_dir):
    # Load PNGs that begin with "step_" in the config_dir directory
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(".png") and f.startswith("step_")]
    )
    # Get step numbers from the filenames
    steps = [int(f.split("_")[-1].split(".")[0]) for f in image_files]

    img = widgets.Image(format='png')

    steps2index = {s: i for i, s in enumerate(steps)}
    # Slider shows actual step numbers
    stepper = widgets.BoundedIntText(value=steps[0], min=min(steps), max=max(steps), step=steps[1] - steps[0], description="Step:")

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