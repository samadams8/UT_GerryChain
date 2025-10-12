#!/usr/bin/env python3
"""
Utah block dataset compilation for GerryChain analysis

This script combines:
- Election data (2016, 2018, 2020, 2024) at census block level
- Demographic data (race, ethnicity)
- County and municipality assignments
- Communities of Interest (COI) assignments
- All data formatted for GerryChain compatibility

Save to data/UT_blocks.geojson

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

def load_census_demographics():
    """Load and process census demographic data from PL94-171 tables."""
    print("Loading census demographic data...")
    
    # Load P1 (total population) data
    p1_path = "data/raw/ut_pl2020_b/ut_pl2020_p1_b.shp"
    p1 = gpd.read_file(p1_path)
    
    # Load P2 (race) data  
    p2_path = "data/raw/ut_pl2020_b/ut_pl2020_p2_b.shp"
    p2 = gpd.read_file(p2_path)
    
    # Load P3 (ethnicity) data
    p3_path = "data/raw/ut_pl2020_b/ut_pl2020_p3_b.shp"
    p3 = gpd.read_file(p3_path)
    
    # Load P4 (voting age population) data
    p4_path = "data/raw/ut_pl2020_b/ut_pl2020_p4_b.shp"
    p4 = gpd.read_file(p4_path)
    
    # Start with P1 as base (contains GEOID20 and total population)
    blocks = p1[['GEOID20', 'geometry']].copy()
    
    # Add total population from P1
    blocks['TOTPOP'] = p1['P0010001']  # Total population
    
    # Add voting age population from P4
    blocks['VAP'] = p4['P0040001']  # Voting age population
    
    # Add race/ethnicity data from P2 and P3
    # Non-Hispanic categories
    blocks['NH_WHITE'] = p2['P0020002']  # Non-Hispanic White alone
    blocks['NH_BLACK'] = p2['P0020003']  # Non-Hispanic Black alone
    blocks['NH_AMIN'] = p2['P0020004']   # Non-Hispanic American Indian/Alaska Native alone
    blocks['NH_ASIAN'] = p2['P0020005']  # Non-Hispanic Asian alone
    blocks['NH_NHPI'] = p2['P0020006']   # Non-Hispanic Native Hawaiian/Pacific Islander alone
    blocks['NH_OTHER'] = p2['P0020007']  # Non-Hispanic Some Other Race alone
    blocks['NH_2MORE'] = p2['P0020008']  # Non-Hispanic Two or More Races
    
    # Hispanic categories
    blocks['HISP'] = p3['P0030002']      # Hispanic or Latino
    blocks['H_WHITE'] = p3['P0030003']   # Hispanic White alone
    blocks['H_BLACK'] = p3['P0030004']   # Hispanic Black alone
    blocks['H_AMIN'] = p3['P0030005']    # Hispanic American Indian/Alaska Native alone
    blocks['H_ASIAN'] = p3['P0030006']   # Hispanic Asian alone
    blocks['H_NHPI'] = p3['P0030007']    # Hispanic Native Hawaiian/Pacific Islander alone
    blocks['H_OTHER'] = p3['P0030008']   # Hispanic Some Other Race alone
    blocks['H_2MORE'] = p3['P0030009']   # Hispanic Two or More Races
    
    return blocks

def load_election_data():
    """Load and process election data from disaggregated files."""
    print("Loading election data...")
    
    # Define the offices we want to include
    target_offices = ['PRE', 'GOV', 'ATG', 'AUD', 'TRE', 'USS']
    
    election_years = [2016, 2018, 2020, 2024]
    election_data = {}
    
    for year in election_years:
        print(f"  Processing {year} election data...")
        disagg_path = f"data/disagg/ut_{year}_gen_2020_blocks/ut_{year}_gen_2020_blocks.shp"
        
        if os.path.exists(disagg_path):
            year_data = gpd.read_file(disagg_path)
            
            # Start with GEOID20
            year_dict = {'GEOID20': year_data['GEOID20']}
            
            # Get all election columns and filter by office
            election_cols = [col for col in year_data.columns if col.startswith('G')]
            
            # Group by office and aggregate by party
            for office in target_offices:
                office_cols = [col for col in election_cols if col[3:6] == office]
                
                if office_cols:
                    # Initialize party totals
                    republican_votes = year_data[office_cols[0]] * 0  # Start with zeros
                    democrat_votes = year_data[office_cols[0]] * 0
                    other_votes = year_data[office_cols[0]] * 0
                    
                    # Aggregate by party
                    for col in office_cols:
                        if len(col) >= 7:
                            party = col[6]  # Extract party code (R, D, etc.)
                            if party == 'R':
                                republican_votes += year_data[col].fillna(0)
                            elif party == 'D':
                                democrat_votes += year_data[col].fillna(0)
                            else:
                                other_votes += year_data[col].fillna(0)
                    
                    # Create aggregated columns using YYOOOP format
                    year_suffix = str(year)[-2:]  # Two digit year
                    year_dict[f"{year_suffix}{office}R"] = republican_votes
                    year_dict[f"{year_suffix}{office}D"] = democrat_votes
                    year_dict[f"{year_suffix}{office}O"] = other_votes
            
            election_data[year] = year_dict
            print(f"    Found {len([k for k in year_dict.keys() if k != 'GEOID20'])} aggregated election columns")
        else:
            print(f"    Warning: {disagg_path} not found")
    
    return election_data

def load_coi_data():
    """Load and process Communities of Interest data."""
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
    """Load municipality and county assignment data."""
    print("Loading municipality and county data...")
    
    # Municipalities
    muni_path = "data/cois/UtahMunicipalBoundaries/Municipalities.shp"
    if os.path.exists(muni_path):
        municipalities = gpd.read_file(muni_path)
        return municipalities
    else:
        print("    Warning: Municipalities file not found")
        return None

def assign_coi_to_blocks(blocks, coi_data):
    """Assign Communities of Interest to blocks using spatial operations."""
    print("Assigning Communities of Interest to blocks...")
    
    # Initialize COI columns with blank string (not assigned)
    coi_columns = ['HIGHERED_ID', 'METRO_ID', 'SCHDIST_ID', 'BASIN_ID', 'WATER_ID']
    for col in coi_columns:
        blocks[col] = ''
    
    # Use maup to assign COI to blocks
    for coi_name, coi_gdf in coi_data.items():
        if coi_gdf is not None:
            print(f"  Assigning {coi_name}...")
            try:
                # Ensure both datasets have the same CRS
                if coi_gdf.crs != blocks.crs:
                    print(f"    Converting {coi_name} CRS from {coi_gdf.crs} to {blocks.crs}")
                    coi_gdf = coi_gdf.to_crs(blocks.crs)
                
                # Use maup.assign to assign COI to blocks
                assignment = maup.assign(blocks, coi_gdf)
                
                # Create a mask for blocks that were assigned
                assigned_mask = assignment.notna()
                
                # Set assigned blocks to the COI ID
                if assigned_mask.any():
                    coi_ids = coi_gdf.iloc[assignment[assigned_mask].astype(int)][coi_name].values
                    blocks.loc[assigned_mask, coi_name] = coi_ids
                
                print(f"    Assigned {assigned_mask.sum()} blocks to {coi_name}")
                
            except Exception as e:
                print(f"    Error assigning {coi_name}: {e}")
                # Set all to blank string (not assigned) if there's an error
                blocks[coi_name] = ''
    
    return blocks

def assign_municipalities_to_blocks(blocks, municipalities):
    """Assign municipalities to blocks."""
    print("Assigning municipalities to blocks...")
    
    # Initialize municipality columns
    blocks['MUNINAME'] = ''
    blocks['MUNIID'] = ''
    
    if municipalities is not None:
        try:
            # Ensure both datasets have the same CRS
            if municipalities.crs != blocks.crs:
                print(f"    Converting municipalities CRS from {municipalities.crs} to {blocks.crs}")
                municipalities = municipalities.to_crs(blocks.crs)
            
            # Use maup to assign municipalities to blocks
            assignment = maup.assign(blocks, municipalities)
            
            # Get municipality names and IDs for assigned blocks
            assigned_mask = assignment.notna()
            if assigned_mask.any():
                # Get the municipality data for assigned blocks
                muni_data = municipalities.iloc[assignment[assigned_mask].astype(int)]
                
                # Assign municipality names and IDs
                blocks.loc[assigned_mask, 'MUNINAME'] = muni_data['NAME'].values
                blocks.loc[assigned_mask, 'MUNIID'] = muni_data.index.values
                
                print(f"    Assigned {assigned_mask.sum()} blocks to municipalities")
            else:
                print("    No blocks assigned to municipalities")
                
        except Exception as e:
            print(f"    Error assigning municipalities: {e}")
    
    return blocks

def add_county_info(blocks):
    """Add county information from GEOID20."""
    print("Adding county information...")
    
    # Extract county FIPS from GEOID20 (characters 3-5)
    blocks['COUNTYID'] = blocks['GEOID20'].str[2:5]
    
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
    
    blocks['COUNTYNAME'] = blocks['COUNTYID'].map(utah_counties)
    
    return blocks

def combine_election_data(blocks, election_data):
    """Combine election data from all years into the blocks dataframe."""
    print("Combining election data...")
    
    for year, year_data in election_data.items():
        print(f"  Adding {year} election data...")
        
        # Create a dataframe from the year's data
        year_df = pd.DataFrame(year_data)
        
        # Merge with blocks on GEOID20
        blocks = blocks.merge(year_df, on='GEOID20', how='left', suffixes=('', f'_{year}'))
    
    return blocks

def main():
    """Main function to compile all Utah block data."""
    print("Starting Utah block dataset compilation...")
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load census demographic data
    blocks = load_census_demographics()
    print(f"Loaded {len(blocks)} census blocks")
    
    # Load election data
    election_data = load_election_data()
    
    # Load COI data
    coi_data = load_coi_data()
    
    # Load municipality data
    municipalities = load_municipality_data()
    
    # Assign COI to blocks
    blocks = assign_coi_to_blocks(blocks, coi_data)
    
    # Assign municipalities to blocks
    blocks = assign_municipalities_to_blocks(blocks, municipalities)
    
    # Add county information
    blocks = add_county_info(blocks)
    
    # Combine election data
    blocks = combine_election_data(blocks, election_data)
    
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
        if col not in blocks.columns:
            blocks[col] = default_val
    
    # Fill NaN values with appropriate defaults
    blocks = blocks.fillna({
        'MUNINAME': '', 'MUNIID': '', 'COUNTYNAME': '', 'COUNTYID': '',
        'TOTPOP': 0, 'VAP': 0
    })
    
    # Fill NaN values for numeric columns with 0
    numeric_columns = blocks.select_dtypes(include=[np.number]).columns
    blocks[numeric_columns] = blocks[numeric_columns].fillna(0)
    
    # Save to GeoJSON
    output_path = 'data/UT_blocks.geojson'
    print(f"Saving compiled data to {output_path}...")
    
    # Convert to a projected CRS appropriate for Utah (UTM Zone 12N)
    # This will prevent GerryChain warnings about geographic CRS
    if blocks.crs is None:
        blocks = blocks.set_crs('EPSG:4269')  # NAD83 (original CRS)
    
    # Convert to UTM Zone 12N (EPSG:32612) which is appropriate for Utah
    blocks = blocks.to_crs('EPSG:32612')
    
    blocks.to_file(output_path, driver='GeoJSON')
    
    print(f"Compilation complete! Saved {len(blocks)} blocks to {output_path}")
    print(f"Columns: {list(blocks.columns)}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total blocks: {len(blocks)}")
    print(f"Total population: {blocks['TOTPOP'].sum():,}")
    print(f"Total VAP: {blocks['VAP'].sum():,}")
    print(f"Blocks with municipalities: {(blocks['MUNINAME'] != '').sum()}")
    print(f"Blocks in higher education areas: {(blocks['HIGHERED_ID'] != '').sum()}")
    print(f"Blocks in metro areas: {(blocks['METRO_ID'] != '').sum()}")
    print(f"Blocks in school districts: {(blocks['SCHDIST_ID'] != '').sum()}")

if __name__ == "__main__":
    main()
