#!/usr/bin/env python3
"""
Prepare transitability datasets by filtering and processing geographic data.

This script processes the large geographic datasets to create filtered versions
suitable for transitability analysis in redistricting.

Usage:
    python 05_prepare_transitability_data.py

Outputs:
    - data/geography_processed/UtahMajorLakes_filtered.shp
    - data/geography_processed/UtahMajorRivers_filtered.shp  
    - data/geography_processed/UtahRoads_filtered.shp
    - data/geography_processed/processing_summary.txt
"""

import os
import geopandas as gpd
import pandas as pd
from pathlib import Path
import time

def create_output_dir():
    """Create the processed data directory."""
    output_dir = Path("data/geography_processed")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def filter_water_bodies(output_dir):
    """Filter lakes and rivers to major water bodies only."""
    print("=== FILTERING WATER BODIES ===")
    
    # Load lakes
    print("Loading lakes...")
    lakes = gpd.read_file("data/geography/UtahMajorLakes/UtahMajorLakes.shp")
    print(f"  Original: {len(lakes):,} features")
    
    # Filter to major lakes (> 1 sq km)
    major_lakes = lakes[lakes['SQ_KM'] > 1.0]
    print(f"  Filtered: {len(major_lakes):,} features ({len(major_lakes)/len(lakes)*100:.1f}%)")
    
    # Save filtered lakes
    lakes_output = output_dir / "UtahMajorLakes_filtered.shp"
    major_lakes.to_file(lakes_output)
    print(f"  Saved: {lakes_output}")
    
    # Load rivers
    print("\nLoading rivers...")
    rivers = gpd.read_file("data/geography/UtahMajorRiversPoly/UtahMajorRiversPoly.shp")
    print(f"  Original: {len(rivers):,} features")
    
    # Filter to major rivers (> 0.5 sq km)
    major_rivers = rivers[rivers['SQ_KM'] > 0.5]
    print(f"  Filtered: {len(major_rivers):,} features ({len(major_rivers)/len(rivers)*100:.1f}%)")
    
    # Save filtered rivers
    rivers_output = output_dir / "UtahMajorRivers_filtered.shp"
    major_rivers.to_file(rivers_output)
    print(f"  Saved: {rivers_output}")
    
    return major_lakes, major_rivers

def filter_roads(output_dir):
    """Filter roads to exclude local streets (CARTOCODE 11)."""
    print("\n=== FILTERING ROADS ===")
    
    print("Loading roads...")
    roads = gpd.read_file("data/geography/UtahRoads/Roads.shp")
    print(f"  Original: {len(roads):,} features")
    
    # Filter to exclude local streets (CARTOCODE 11)
    # This gives us major highways, state routes, collectors, and arterials
    filtered_roads = roads[roads['CARTOCODE'] != '11']
    print(f"  Filtered: {len(filtered_roads):,} features ({len(filtered_roads)/len(roads)*100:.1f}%)")
    
    # Show breakdown by CARTOCODE
    print("\n  Filtered roads by type:")
    cartocode_counts = filtered_roads['CARTOCODE'].value_counts().sort_index()
    for code, count in cartocode_counts.items():
        pct = count / len(filtered_roads) * 100
        print(f"    CARTOCODE {code}: {count:,} ({pct:.1f}%)")
    
    # Save filtered roads
    roads_output = output_dir / "UtahRoads_filtered.shp"
    filtered_roads.to_file(roads_output)
    print(f"\n  Saved: {roads_output}")
    
    return filtered_roads

def generate_summary(output_dir, major_lakes, major_rivers, filtered_roads):
    """Generate processing summary."""
    print("\n=== GENERATING SUMMARY ===")
    
    summary_lines = [
        "TRANSITABILITY DATA PROCESSING SUMMARY",
        "=" * 50,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "WATER BODIES:",
        f"  Major Lakes: {len(major_lakes):,} features",
        f"  Major Rivers: {len(major_rivers):,} features", 
        f"  Total Water Bodies: {len(major_lakes) + len(major_rivers):,} features",
        "",
        "ROADS:",
        f"  Filtered Roads: {len(filtered_roads):,} features",
        f"  Excluded: Local streets (CARTOCODE 11)",
        "",
        "FILE SIZES:",
    ]
    
    # Calculate file sizes
    for file_path in output_dir.glob("*.shp"):
        size_mb = sum(f.stat().st_size for f in file_path.parent.glob(f"{file_path.stem}.*")) / (1024*1024)
        summary_lines.append(f"  {file_path.name}: {size_mb:.1f} MB")
    
    total_size = sum(f.stat().st_size for f in output_dir.glob("*")) / (1024*1024)
    summary_lines.extend([
        f"  Total: {total_size:.1f} MB",
        "",
        "USAGE:",
        "  These filtered datasets are used by utgc/transitability.py",
        "  to create transitability-aware graph connectivity.",
        "",
        "CONFIGURATION:",
        "  - Lakes: > 1 sq km (major water bodies only)",
        "  - Rivers: > 0.5 sq km (wide river segments only)", 
        "  - Roads: Exclude CARTOCODE 11 (local streets)",
        "  - Orphaned precincts handled via hierarchical fallback"
    ])
    
    summary_text = "\n".join(summary_lines)
    
    # Save summary
    summary_path = output_dir / "processing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"Summary saved: {summary_path}")
    print("\n" + summary_text)

def main():
    """Main processing function."""
    print("TRANSITABILITY DATA PREPARATION")
    print("=" * 50)
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")
    
    # Process datasets
    major_lakes, major_rivers = filter_water_bodies(output_dir)
    filtered_roads = filter_roads(output_dir)
    
    # Generate summary
    generate_summary(output_dir, major_lakes, major_rivers, filtered_roads)
    
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE!")
    print(f"Filtered datasets saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Implement utgc/transitability.py module")
    print("2. Integrate into utgc/build.py")
    print("3. Add configuration to 03_configure_sampling.ipynb")

if __name__ == "__main__":
    main()
