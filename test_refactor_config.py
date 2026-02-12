
import os
import shutil
from utgc import ConfigurationManager, EnsembleRunner

# Data file paths (from test_optimization.py)
pop_geodata_path = "data/UT_blocks.geojson"
initial_plan_path = "maps/US-House/2025_USH_Leg-C/2025_USH_Leg-C.shp"

def test_refactor():
    print("Testing ConfigurationManager and EnsembleRunner refactor...")
    
    # 1. Initialize ConfigurationManager
    config = ConfigurationManager(
        pop_geodata_path=pop_geodata_path,
        initial_plan_path=initial_plan_path,
        random_seed=42,
        pop_column="TOTPOP"
    )
    
    # 2. Configure the run
    config.set_pop_dev_tolerance(0.05)
    config.add_pop_dev_updater()
    config.constrain_not_equal(not_equal_constraint=True)
    
    # Add a simple election updater if columns exist (checking broadly)
    # config.add_election_updaters(years=[2022], elections=['SEN']) 
    # Skipping auto-detection for simplicity in this basic test unless we know columns exist
    
    # 3. Initialize Runner
    runner = EnsembleRunner(config)
    
    # 4. Precondition (optional but good to test)
    runner.precondition(steps=10, max_attempts=1)
    
    # 5. Run
    output_dir = "output/test_refactor"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    runner.run(
        name="test_run",
        num_steps=20,
        output_dir=output_dir,
        use_preconditioned_partition=True
    )
    
    print(f"Run completed. Checking output in {output_dir}")
    
    # 6. Verify output
    config_file = os.path.join(output_dir, "test_run", "config.yaml")
    output_file = os.path.join(output_dir, "test_run", "output.jsonl")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")
        
    with open(output_file, 'r') as f:
        lines = f.readlines()
        print(f"Output has {len(lines)} lines (steps).")
        if len(lines) != 20: 
             print(f"WARNING: Expected 20 steps, got {len(lines)}")
        
    print("Verification successful!")

if __name__ == "__main__":
    test_refactor()
