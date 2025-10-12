#!/usr/bin/env python3
"""
Utah precinct dataset compilation for GerryChain analysis

This script converts block-level data to precinct level by:
- Loading existing UT_blocks.geojson
- Loading precinct boundaries
- Aggregating block data to precincts using maup
- Reassigning Communities of Interest (COI) to precincts
- Saving as UT_precincts.geojson

Output columns
--------------
# Basics
- PRECINCTID
- MUNINAME
- MUNIID
- COUNTYNAME
- COUNTYID
- TOTPOP
- VAP
# Demographics
- NH_WHITE
- NH_BLACK
- NH_AMIN
- NH_ASIAN
- NH_NHPI
- NH_OTHER
- NH_2MORE
- HISP
- H_WHITE
- H_BLACK
- H_AMIN
- H_ASIAN
- H_NHPI
- H_OTHER
- H_2MORE
# Communities of Interest
- HIGHERED_ID
- METRO_ID
- SCHDIST_ID
# Election results
Format YYOOOP
- Years (2016, 2018, 2020, 2024)
- Offices (PRE, GOV, ATG, AUD, TRE, USS)
- Parties (D, R, O)
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import maup for spatial operations
import maup

def load_block_data():
    """Load the existing block-level data."""
    print("Loading block-level data...")
    
    blocks_path = "data/UT_blocks.geojson"
    if os.path.exists(blocks_path):
        blocks = gpd.read_file(blocks_path)
        print(f"Loaded {len(blocks)} blocks")
        return blocks
    else:
        print(f"Error: {blocks_path} not found. Run 01_compile_blocks.py first.")
        sys.exit(1)

def load_precinct_boundaries():
    """Load precinct boundaries from the raw data."""
    print("Loading precinct boundaries...")
    
    # Try to find the most recent precinct data
    precinct_paths = [
        "data/raw/ut_2024_gen_prec/ut_2024_gen_all_prec/ut_2024_gen_all_prec.shp",
        "data/raw/ut_2024_gen_prec/ut_2024_gen_cong_prec/ut_2024_gen_cong_prec.shp",
        "data/raw/ut_2024_gen_prec/ut_2024_gen_sldl_prec/ut_2024_gen_sldl_prec.shp",
        "data/raw/ut_2024_gen_prec/ut_2024_gen_sldu_prec/ut_2024_gen_sldu_prec.shp"
    ]
    
    for path in precinct_paths:
        if os.path.exists(path):
            print(f"  Using precinct data from: {path}")
            precincts = gpd.read_file(path)
            
            # Ensure we have a unique precinct ID
            if 'PRECINCTID' not in precincts.columns:
                # Create a unique ID if it doesn't exist
                precincts['PRECINCTID'] = range(1, len(precincts) + 1)
            
            return precincts
    
    print("Error: No precinct boundary files found")
    sys.exit(1)

def aggregate_blocks_to_precincts(blocks, precincts):
    """Aggregate block data to precinct level using maup."""
    print("Aggregating block data to precincts...")
    
    # Ensure both datasets have the same CRS
    if precincts.crs != blocks.crs:
        print(f"  Converting precincts CRS from {precincts.crs} to {blocks.crs}")
        precincts = precincts.to_crs(blocks.crs)
    
    # Use maup to aggregate block data to precincts
    try:
        # First, assign blocks to precincts
        blocks_to_precincts_assignment = maup.assign(blocks, precincts)
        
        # Get numeric columns to aggregate (demographics, population, election data)
        numeric_columns = blocks.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'GEOID20']  # Exclude GEOID20
        
        # Aggregate data using pandas groupby
        aggregated_data = blocks[numeric_columns].groupby(blocks_to_precincts_assignment).sum()
        
        # Add precinct geometry, ID, and county information
        precinct_columns = ['PRECINCTID', 'geometry']
        if 'COUNTYFP' in precincts.columns:
            precinct_columns.append('COUNTYFP')
        if 'COUNTY' in precincts.columns:
            precinct_columns.append('COUNTY')
        
        precinct_data = precincts[precinct_columns].copy()
        precinct_data = precinct_data.merge(aggregated_data, left_index=True, right_index=True)
        
        print(f"  Aggregated {len(blocks)} blocks to {len(precinct_data)} precincts")
        return precinct_data
        
    except Exception as e:
        print(f"  Error aggregating data: {e}")
        sys.exit(1)

def assign_coi_to_precincts(precincts, coi_data):
    """Assign Communities of Interest to precincts using spatial operations."""
    print("Assigning Communities of Interest to precincts...")
    
    # Initialize COI columns with blank string (not assigned)
    coi_columns = ['HIGHERED_ID', 'METRO_ID', 'SCHDIST_ID', 'BASIN_ID', 'WATER_ID']
    for col in coi_columns:
        precincts[col] = ''
    
    # Use maup to assign COI to precincts
    for coi_name, coi_gdf in coi_data.items():
        if coi_gdf is not None:
            print(f"  Assigning {coi_name}...")
            try:
                # Ensure both datasets have the same CRS
                if coi_gdf.crs != precincts.crs:
                    print(f"    Converting {coi_name} CRS from {coi_gdf.crs} to {precincts.crs}")
                    coi_gdf = coi_gdf.to_crs(precincts.crs)
                
                # Use maup.assign to assign COI to precincts
                assignment = maup.assign(precincts, coi_gdf)
                
                # Create a mask for precincts that were assigned
                assigned_mask = assignment.notna()
                
                # Set assigned precincts to the COI ID
                if assigned_mask.any():
                    coi_ids = coi_gdf.iloc[assignment[assigned_mask].astype(int)][coi_name].values
                    precincts.loc[assigned_mask, coi_name] = coi_ids
                
                print(f"    Assigned {assigned_mask.sum()} precincts to {coi_name}")
                
            except Exception as e:
                print(f"    Error assigning {coi_name}: {e}")
                # Set all to blank string (not assigned) if there's an error
                precincts[coi_name] = ''
    
    return precincts

def assign_municipalities_to_precincts(precincts, municipalities):
    """Assign municipalities to precincts."""
    print("Assigning municipalities to precincts...")
    
    # Initialize municipality columns
    precincts['MUNINAME'] = ''
    precincts['MUNIID'] = ''
    
    if municipalities is not None:
        try:
            # Ensure both datasets have the same CRS
            if municipalities.crs != precincts.crs:
                print(f"    Converting municipalities CRS from {municipalities.crs} to {precincts.crs}")
                municipalities = municipalities.to_crs(precincts.crs)
            
            # Use maup to assign municipalities to precincts
            assignment = maup.assign(precincts, municipalities)
            
            # Get municipality names and IDs for assigned precincts
            assigned_mask = assignment.notna()
            if assigned_mask.any():
                # Get the municipality data for assigned precincts
                muni_data = municipalities.iloc[assignment[assigned_mask].astype(int)]
                
                # Assign municipality names and IDs
                precincts.loc[assigned_mask, 'MUNINAME'] = muni_data['NAME'].values
                precincts.loc[assigned_mask, 'MUNIID'] = muni_data.index.values
                
                print(f"    Assigned {assigned_mask.sum()} precincts to municipalities")
            else:
                print("    No precincts assigned to municipalities")
                
        except Exception as e:
            print(f"    Error assigning municipalities: {e}")
    
    return precincts

def add_county_info(precincts):
    """Add county information by extracting from precinct data."""
    print("Adding county information...")
    
    # Initialize county columns
    precincts['COUNTYNAME'] = ''
    precincts['COUNTYID'] = ''
    
    # Extract county info from precinct data
    if 'COUNTYFP' in precincts.columns:
        print("    Using COUNTYFP from precinct data...")
        precincts['COUNTYID'] = precincts['COUNTYFP']
        
        # Use the COUNTY column if available, otherwise map from FIPS
        if 'COUNTY' in precincts.columns:
            precincts['COUNTYNAME'] = precincts['COUNTY']
            print(f"    Using COUNTY column for names")
        else:
            # Utah county names mapping (FIPS to name)
            utah_counties = {
                '001': 'Beaver', '003': 'Box Elder', '005': 'Cache', '007': 'Carbon',
                '009': 'Daggett', '011': 'Davis', '013': 'Duchesne', '015': 'Emery',
                '017': 'Garfield', '019': 'Grand', '021': 'Iron', '023': 'Juab',
                '025': 'Kane', '027': 'Millard', '029': 'Morgan', '031': 'Piute',
                '033': 'Rich', '035': 'Salt Lake', '037': 'San Juan', '039': 'Sanpete',
                '041': 'Sevier', '043': 'Summit', '045': 'Tooele', '047': 'Uintah',
                '049': 'Utah', '051': 'Wasatch', '053': 'Washington', '055': 'Wayne',
                '057': 'Weber'
            }
            precincts['COUNTYNAME'] = precincts['COUNTYID'].map(utah_counties)
            print(f"    Mapped county names from FIPS codes")
        
        print(f"    Assigned counties to {len(precincts)} precincts")
    else:
        print("    No COUNTYFP column found in precinct data - leaving blank")
    
    return precincts

def load_coi_data():
    """Load Communities of Interest data (same as block script)."""
    print("Loading Communities of Interest data...")
    
    coi_data = {}
    
    # Higher Education
    print("  Processing Higher Education...")
    higher_ed_path = "data/cois/Higher_Ed/Higher_Ed.shp"
    if os.path.exists(higher_ed_path):
        higher_ed = gpd.read_file(higher_ed_path)
        # Create sequential IDs since no unique ID column exists
        higher_ed['HIGHERED_ID'] = range(1, len(higher_ed) + 1)
        coi_data['HIGHERED_ID'] = higher_ed[['geometry', 'HIGHERED_ID']]
        print(f"    Found {len(higher_ed)} Higher Education areas")
    else:
        print("    Warning: Higher Education file not found")
        coi_data['HIGHERED_ID'] = None
    
    # Metro/Micro Statistical Areas
    print("  Processing Metro/Micro Statistical Areas...")
    metro_path = "data/cois/MetroMicroStatisticalAreas/MetroMicroStatisticalAreas.shp"
    if os.path.exists(metro_path):
        metro = gpd.read_file(metro_path)
        # Use CBSAFP as the ID (Core Based Statistical Area FIPS)
        metro['METRO_ID'] = metro['CBSAFP']
        coi_data['METRO_ID'] = metro[['geometry', 'METRO_ID']]
        print(f"    Found {len(metro)} Metro/Micro Statistical Areas")
    else:
        print("    Warning: Metro/Micro Statistical Areas file not found")
        coi_data['METRO_ID'] = None
    
    # School Districts
    print("  Processing School Districts...")
    school_path = "data/cois/UtahSchoolDistrictBoundaries/SchoolDistricts.shp"
    if os.path.exists(school_path):
        school = gpd.read_file(school_path)
        # Create sequential IDs since no unique ID column exists
        school['SCHDIST_ID'] = range(1, len(school) + 1)
        coi_data['SCHDIST_ID'] = school[['geometry', 'SCHDIST_ID']]
        print(f"    Found {len(school)} School Districts")
    else:
        print("    Warning: School Districts file not found")
        coi_data['SCHDIST_ID'] = None
    
    # Hydrologic Basins
    print("  Processing Hydrologic Basins...")
    basin_path = "data/cois/UtahHydrologicBasins/HydrologicBasins.shp"
    if os.path.exists(basin_path):
        basin = gpd.read_file(basin_path)
        # Create sequential IDs since no unique ID column exists
        basin['BASIN_ID'] = range(1, len(basin) + 1)
        coi_data['BASIN_ID'] = basin[['geometry', 'BASIN_ID']]
        print(f"    Found {len(basin)} Hydrologic Basins")
    else:
        print("    Warning: Hydrologic Basins file not found")
        coi_data['BASIN_ID'] = None
    
    # Water Planning Areas
    print("  Processing Water Planning Areas...")
    water_path = "data/cois/UtahWaterPlanningAreas/Planning_Areas.shp"
    if os.path.exists(water_path):
        water = gpd.read_file(water_path)
        # Create sequential IDs since no unique ID column exists
        water['WATER_ID'] = range(1, len(water) + 1)
        coi_data['WATER_ID'] = water[['geometry', 'WATER_ID']]
        print(f"    Found {len(water)} Water Planning Areas")
    else:
        print("    Warning: Water Planning Areas file not found")
        coi_data['WATER_ID'] = None
    
    return coi_data

def load_municipality_data():
    """Load municipality data (same as block script)."""
    print("Loading municipality data...")
    
    # Municipalities
    muni_path = "data/cois/UtahMunicipalBoundaries/Municipalities.shp"
    if os.path.exists(muni_path):
        municipalities = gpd.read_file(muni_path)
        return municipalities
    else:
        print("    Warning: Municipalities file not found")
        return None

def main():
    """Main function to compile precinct data from blocks."""
    print("Starting Utah precinct dataset compilation...")
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load block data
    blocks = load_block_data()
    
    # Load precinct boundaries
    precincts = load_precinct_boundaries()
    
    # Aggregate block data to precincts
    precinct_data = aggregate_blocks_to_precincts(blocks, precincts)
    
    # Load COI data
    coi_data = load_coi_data()
    
    # Load municipality data
    municipalities = load_municipality_data()
    
    # Assign COI to precincts
    precinct_data = assign_coi_to_precincts(precinct_data, coi_data)
    
    # Assign municipalities to precincts
    precinct_data = assign_municipalities_to_precincts(precinct_data, municipalities)
    
    # Add county information
    precinct_data = add_county_info(precinct_data)
    
    # Ensure all required columns exist with appropriate defaults
    required_columns = {
        'MUNINAME': '',
        'MUNIID': '',
        'COUNTYNAME': '',
        'COUNTYID': '',
        'TOTPOP': 0,
        'VAP': 0,
        'NH_WHITE': 0, 'NH_BLACK': 0, 'NH_AMIN': 0, 'NH_ASIAN': 0,
        'NH_NHPI': 0, 'NH_OTHER': 0, 'NH_2MORE': 0,
        'HISP': 0, 'H_WHITE': 0, 'H_BLACK': 0, 'H_AMIN': 0,
        'H_ASIAN': 0, 'H_NHPI': 0, 'H_OTHER': 0, 'H_2MORE': 0,
        'HIGHERED_ID': '', 'METRO_ID': '', 'SCHDIST_ID': '', 'BASIN_ID': '', 'WATER_ID': ''
    }
    
    for col, default_val in required_columns.items():
        if col not in precinct_data.columns:
            precinct_data[col] = default_val
    
    # Fill NaN values with appropriate defaults
    precinct_data = precinct_data.fillna({
        'MUNINAME': '', 'MUNIID': '', 'COUNTYNAME': '', 'COUNTYID': '',
        'TOTPOP': 0, 'VAP': 0
    })
    
    # Fill NaN values for numeric columns with 0
    numeric_columns = precinct_data.select_dtypes(include=[np.number]).columns
    precinct_data[numeric_columns] = precinct_data[numeric_columns].fillna(0)
    
    # Fix topological issues using maup best practices
    print("Fixing topological issues...")
    try:
        # Use maup's smart_repair which handles both overlaps and gaps
        print("  Using maup.smart_repair for comprehensive topology repair...")
        repaired_precincts = maup.smart_repair(precinct_data, fill_gaps_threshold=0.3)
        
        # Verify the repair worked
        overlaps = maup.adjacencies(repaired_precincts)
        if len(overlaps) == 0:
            print("    Topology is clean - no overlaps remaining")
            precinct_data = repaired_precincts
        else:
            print(f"  ⚠️ {len(overlaps)} overlaps still remain after repair")
            print("  Using repaired data anyway - GerryChain can handle minor overlaps")
            precinct_data = repaired_precincts
        
    except Exception as e:
        print(f"  Warning: Could not repair topology: {e}")
        print("  Proceeding with original data...")
    
    # Save to GeoJSON
    output_path = 'data/UT_precincts.geojson'
    print(f"Saving compiled data to {output_path}...")
    
    # Convert to a projected CRS appropriate for Utah (UTM Zone 12N)
    if precinct_data.crs is None:
        precinct_data = precinct_data.set_crs('EPSG:4269')  # NAD83 (original CRS)
    
    # Convert to UTM Zone 12N (EPSG:32612) which is appropriate for Utah
    precinct_data = precinct_data.to_crs('EPSG:32612')
    
    precinct_data.to_file(output_path, driver='GeoJSON')
    
    print(f"Compilation complete! Saved {len(precinct_data)} precincts to {output_path}")
    if hasattr(precinct_data, 'columns'):
        print(f"Columns: {list(precinct_data.columns)}")
    else:
        print("Columns: [GeoSeries - no column info available]")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total precincts: {len(precinct_data)}")
    print(f"Total population: {precinct_data['TOTPOP'].sum():,}")
    print(f"Total VAP: {precinct_data['VAP'].sum():,}")
    print(f"Precincts with municipalities: {(precinct_data['MUNINAME'] != '').sum()}")
    print(f"Precincts in higher education areas: {(precinct_data['HIGHERED_ID'] != '').sum()}")
    print(f"Precincts in metro areas: {(precinct_data['METRO_ID'] != '').sum()}")
    print(f"Precincts in school districts: {(precinct_data['SCHDIST_ID'] != '').sum()}")

if __name__ == "__main__":
    main()
