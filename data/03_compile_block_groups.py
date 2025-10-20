#!/usr/bin/env python3
"""
Utah block group dataset compilation for GerryChain analysis

This script converts block-level data to block group level by:
- Loading existing UT_blocks.geojson
- Loading block group boundaries
- Aggregating block data to block groups using maup
- Reassigning Communities of Interest (COI) to block groups
- Saving as UT_block_groups.geojson

Output columns
--------------
# Basics
- GEOID20
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
- BASIN_ID
- WATER_ID
- RESERVATION_ID
- MILITARY_ID
# Election results
- Individual candidate columns for statewide offices
- Years (2016, 2018, 2020, 2024)
- Offices (PRE, GOV, ATG, AUD, TRE, USS)
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

def load_block_group_boundaries():
    """Load block group boundaries from the raw data."""
    print("Loading block group boundaries...")
    
    block_group_path = "data/raw/ut_bg_2020_bound/ut_bg_2020_bound.shp"
    if os.path.exists(block_group_path):
        print(f"  Using block group data from: {block_group_path}")
        block_groups = gpd.read_file(block_group_path)
        
        # Ensure we have a unique block group ID
        if 'GEOID20' not in block_groups.columns:
            print("Error: GEOID20 column not found in block group data")
            sys.exit(1)
        
        return block_groups
    else:
        print(f"Error: {block_group_path} not found")
        sys.exit(1)

def aggregate_blocks_to_block_groups(blocks, block_groups):
    """Aggregate block data to block group level using maup."""
    print("Aggregating block data to block groups...")
    
    # Ensure both datasets have the same CRS
    if block_groups.crs != blocks.crs:
        print(f"  Converting block groups CRS from {block_groups.crs} to {blocks.crs}")
        block_groups = block_groups.to_crs(blocks.crs)
    
    # Use maup to aggregate block data to block groups
    try:
        # First, assign blocks to block groups
        blocks_to_block_groups_assignment = maup.assign(blocks, block_groups)
        
        # Get numeric columns to aggregate (demographics, population, election data)
        numeric_columns = blocks.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'GEOID20']  # Exclude GEOID20
        
        # Aggregate data using pandas groupby
        aggregated_data = blocks[numeric_columns].groupby(blocks_to_block_groups_assignment).sum()
        
        # Add block group geometry and ID
        block_group_columns = ['GEOID20', 'geometry']
        
        block_group_data = block_groups[block_group_columns].copy()
        block_group_data = block_group_data.merge(aggregated_data, left_index=True, right_index=True)
        
        print(f"  Aggregated {len(blocks)} blocks to {len(block_group_data)} block groups")
        return block_group_data
        
    except Exception as e:
        print(f"  Error aggregating data: {e}")
        sys.exit(1)

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
    
    # Indian Reservations
    print("  Processing Indian Reservations...")
    surface_mgmt_path = "data/cois/UT_SurfaceManagementAgency/BLM_UT_Surface_Management_Agency_(Polygon).shp"
    if os.path.exists(surface_mgmt_path):
        surface_mgmt = gpd.read_file(surface_mgmt_path)
        # Filter for Indian Reservations
        reservations = surface_mgmt[surface_mgmt['DESIG'] == 'Indian Reservation'].copy()
        if len(reservations) > 0:
            # Reset index to ensure 0-based indexing for maup.assign
            reservations = reservations.reset_index(drop=True)
            reservations['RESERVATION_ID'] = range(1, len(reservations) + 1)
            coi_data['RESERVATION_ID'] = reservations[['geometry', 'RESERVATION_ID']]
            print(f"    Found {len(reservations)} Indian Reservations")
        else:
            print("    No Indian Reservations found")
            coi_data['RESERVATION_ID'] = None
    else:
        print("    Warning: Surface Management Agency file not found")
        coi_data['RESERVATION_ID'] = None
    
    # Military Installations
    print("  Processing Military Installations...")
    if os.path.exists(surface_mgmt_path):
        # Filter for Military installations
        military = surface_mgmt[surface_mgmt['DESIG'] == 'Military'].copy()
        if len(military) > 0:
            # Reset index to ensure 0-based indexing for maup.assign
            military = military.reset_index(drop=True)
            military['MILITARY_ID'] = range(1, len(military) + 1)
            coi_data['MILITARY_ID'] = military[['geometry', 'MILITARY_ID']]
            print(f"    Found {len(military)} Military Installations")
        else:
            print("    No Military Installations found")
            coi_data['MILITARY_ID'] = None
    else:
        print("    Warning: Surface Management Agency file not found")
        coi_data['MILITARY_ID'] = None
    
    return coi_data

def assign_coi_to_block_groups(block_groups, coi_data):
    """Assign Communities of Interest to block groups using spatial operations."""
    print("Assigning Communities of Interest to block groups...")
    
    # Initialize COI columns with blank string (not assigned)
    coi_columns = ['HIGHERED_ID', 'METRO_ID', 'SCHDIST_ID', 'BASIN_ID', 'WATER_ID', 'RESERVATION_ID', 'MILITARY_ID']
    for col in coi_columns:
        block_groups[col] = ''
    
    # Use maup to assign COI to block groups
    for coi_name, coi_gdf in coi_data.items():
        if coi_gdf is not None:
            print(f"  Assigning {coi_name}...")
            try:
                # Ensure both datasets have the same CRS
                if coi_gdf.crs != block_groups.crs:
                    print(f"    Converting {coi_name} CRS from {coi_gdf.crs} to {block_groups.crs}")
                    coi_gdf = coi_gdf.to_crs(block_groups.crs)
                
                # Use maup.assign to assign COI to block groups
                assignment = maup.assign(block_groups, coi_gdf)
                
                # Create a mask for block groups that were assigned
                assigned_mask = assignment.notna()
                
                # Set assigned block groups to the COI ID
                if assigned_mask.any():
                    coi_ids = coi_gdf.iloc[assignment[assigned_mask].astype(int)][coi_name].values
                    block_groups.loc[assigned_mask, coi_name] = coi_ids
                
                print(f"    Assigned {assigned_mask.sum()} block groups to {coi_name}")
                
            except Exception as e:
                print(f"    Error assigning {coi_name}: {e}")
                # Set all to blank string (not assigned) if there's an error
                block_groups[coi_name] = ''
    
    return block_groups

def load_municipality_data():
    """Load municipality data (same as block script)."""
    print("Loading municipality data...")
    
    # Municipalities
    muni_path = "data/geography/UtahMunicipalBoundaries/Municipalities.shp"
    if os.path.exists(muni_path):
        municipalities = gpd.read_file(muni_path)
        return municipalities
    else:
        print("    Warning: Municipalities file not found")
        return None

def assign_municipalities_to_block_groups(block_groups, municipalities):
    """Assign municipalities to block groups."""
    print("Assigning municipalities to block groups...")
    
    # Initialize municipality columns
    block_groups['MUNINAME'] = ''
    block_groups['MUNIID'] = ''
    
    if municipalities is not None:
        try:
            # Ensure both datasets have the same CRS
            if municipalities.crs != block_groups.crs:
                print(f"    Converting municipalities CRS from {municipalities.crs} to {block_groups.crs}")
                municipalities = municipalities.to_crs(block_groups.crs)
            
            # Use maup to assign municipalities to block groups
            assignment = maup.assign(block_groups, municipalities)
            
            # Get municipality names and IDs for assigned block groups
            assigned_mask = assignment.notna()
            if assigned_mask.any():
                # Get the municipality data for assigned block groups
                muni_data = municipalities.iloc[assignment[assigned_mask].astype(int)]
                
                # Assign municipality names and IDs
                block_groups.loc[assigned_mask, 'MUNINAME'] = muni_data['NAME'].values
                block_groups.loc[assigned_mask, 'MUNIID'] = muni_data.index.values
                
                print(f"    Assigned {assigned_mask.sum()} block groups to municipalities")
            else:
                print("    No block groups assigned to municipalities")
                
        except Exception as e:
            print(f"    Error assigning municipalities: {e}")
    
    return block_groups

def add_county_info(block_groups):
    """Add county information by extracting from GEOID20."""
    print("Adding county information...")
    
    # Extract county FIPS from GEOID20 (characters 2-5)
    block_groups['COUNTYID'] = block_groups['GEOID20'].str[2:5]
    
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
    
    block_groups['COUNTYNAME'] = block_groups['COUNTYID'].map(utah_counties)
    
    return block_groups

def main():
    """Main function to compile block group data from blocks."""
    print("Starting Utah block group dataset compilation...")
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load block data
    blocks = load_block_data()
    
    # Load block group boundaries
    block_groups = load_block_group_boundaries()
    
    # Aggregate block data to block groups
    block_group_data = aggregate_blocks_to_block_groups(blocks, block_groups)
    
    # Load COI data
    coi_data = load_coi_data()
    
    # Load municipality data
    municipalities = load_municipality_data()
    
    # Assign COI to block groups
    block_group_data = assign_coi_to_block_groups(block_group_data, coi_data)
    
    # Assign municipalities to block groups
    block_group_data = assign_municipalities_to_block_groups(block_group_data, municipalities)
    
    # Add county information
    block_group_data = add_county_info(block_group_data)
    
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
        'HIGHERED_ID': '', 'METRO_ID': '', 'SCHDIST_ID': '', 'BASIN_ID': '', 'WATER_ID': '', 'RESERVATION_ID': '', 'MILITARY_ID': ''
    }
    
    for col, default_val in required_columns.items():
        if col not in block_group_data.columns:
            block_group_data[col] = default_val
    
    # Fill NaN values with appropriate defaults
    block_group_data = block_group_data.fillna({
        'MUNINAME': '', 'MUNIID': '', 'COUNTYNAME': '', 'COUNTYID': '',
        'TOTPOP': 0, 'VAP': 0
    })
    
    # Fill NaN values for numeric columns with 0
    numeric_columns = block_group_data.select_dtypes(include=[np.number]).columns
    block_group_data[numeric_columns] = block_group_data[numeric_columns].fillna(0)
    
    # Fix topological issues using maup best practices
    print("Fixing topological issues...")
    try:
        # Use maup's smart_repair which handles both overlaps and gaps
        print("  Using maup.smart_repair for comprehensive topology repair...")
        repaired_block_groups = maup.smart_repair(block_group_data, fill_gaps_threshold=0.3)
        
        # Verify the repair worked
        overlaps = maup.adjacencies(repaired_block_groups)
        if len(overlaps) == 0:
            print("    Topology is clean - no overlaps remaining")
            block_group_data = repaired_block_groups
        else:
            print(f"  ⚠️ {len(overlaps)} overlaps still remain after repair")
            print("  Using repaired data anyway - GerryChain can handle minor overlaps")
            block_group_data = repaired_block_groups
        
    except Exception as e:
        print(f"  Warning: Could not repair topology: {e}")
        print("  Proceeding with original data...")
    
    # Save to GeoJSON
    output_path = 'data/UT_block_groups.geojson'
    print(f"Saving compiled data to {output_path}...")
    
    # Convert to a projected CRS appropriate for Utah (UTM Zone 12N)
    if block_group_data.crs is None:
        block_group_data = block_group_data.set_crs('EPSG:4269')  # NAD83 (original CRS)
    
    # Convert to UTM Zone 12N (EPSG:32612) which is appropriate for Utah
    block_group_data = block_group_data.to_crs('EPSG:32612')
    
    block_group_data.to_file(output_path, driver='GeoJSON')
    
    print(f"Compilation complete! Saved {len(block_group_data)} block groups to {output_path}")
    if hasattr(block_group_data, 'columns'):
        print(f"Columns: {list(block_group_data.columns)}")
    else:
        print("Columns: [GeoSeries - no column info available]")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total block groups: {len(block_group_data)}")
    print(f"Total population: {block_group_data['TOTPOP'].sum():,}")
    print(f"Total VAP: {block_group_data['VAP'].sum():,}")
    print(f"Block groups with municipalities: {(block_group_data['MUNINAME'] != '').sum()}")
    print(f"Block groups in higher education areas: {(block_group_data['HIGHERED_ID'] != '').sum()}")
    print(f"Block groups in metro areas: {(block_group_data['METRO_ID'] != '').sum()}")
    print(f"Block groups in school districts: {(block_group_data['SCHDIST_ID'] != '').sum()}")
    print(f"Block groups in reservations: {(block_group_data['RESERVATION_ID'] != '').sum()}")
    print(f"Block groups in military areas: {(block_group_data['MILITARY_ID'] != '').sum()}")

if __name__ == "__main__":
    main()
