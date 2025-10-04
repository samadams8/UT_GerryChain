#!/usr/bin/env python3
"""
Create GeoJSON files combining census shape data with election results and district assignments.

This script creates two GeoJSON files:
1. Block-level data (most granular)
2. Precinct-level data (aggregated from blocks)

Data includes:
- Census demographics (total population, voting age population, race/ethnicity)
- Election results for President, Governor, US Senate, Attorney General, Auditor, Treasurer
- Congressional district assignments from HB2004
- State House and State Senate district assignments from 2024 precinct data
- County and city information
- Republican, Democratic, and Other party vote totals for each office
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from shapely.ops import unary_union
warnings.filterwarnings('ignore')

def load_census_data():
    """Load and combine census demographic data."""
    print("Loading census data...")
    
    # Load P1 (Race) data
    p1 = gpd.read_file("data/raw/ut_pl2020_b/ut_pl2020_p1_b.shp")
    
    # Load P4 (Voting Age) data  
    p4 = gpd.read_file("data/raw/ut_pl2020_b/ut_pl2020_p4_b.shp")
    
    # Combine census data
    census = p1.merge(p4[['GEOID20'] + [col for col in p4.columns if col.startswith('P004')]], 
                     on='GEOID20', how='left')
    
    # Select key demographic columns
    demo_cols = [
        'GEOID20', 'geometry',
        # Total population
        'P0010001',  # Total population
        # Voting age population (18+)
        'P0040001',  # Total 18+
        # Race/ethnicity (P1 table)
        'P0010002',  # Not Hispanic or Latino
        'P0010003',  # Hispanic or Latino
        'P0010004',  # White alone
        'P0010005',  # Black or African American alone
        'P0010006',  # American Indian and Alaska Native alone
        'P0010007',  # Asian alone
        'P0010008',  # Native Hawaiian and Other Pacific Islander alone
        'P0010009',  # Some Other Race alone
        'P0010010',  # Two or More Races
    ]
    
    census = census[demo_cols].copy()
    
    # Add county information
    census['COUNTYFP'] = census['GEOID20'].str[:5]  # Extract county FIPS from GEOID20
    census['STATEFP'] = census['GEOID20'].str[:2]  # Extract state FIPS from GEOID20
    
    # Add area calculation (in square degrees - approximate)
    census['area'] = census.geometry.area
    
    # Create county name mapping
    utah_counties = {
        '49001': 'Beaver County',
        '49003': 'Box Elder County', 
        '49005': 'Cache County',
        '49007': 'Carbon County',
        '49009': 'Daggett County',
        '49011': 'Davis County',
        '49013': 'Duchesne County',
        '49015': 'Emery County',
        '49017': 'Garfield County',
        '49019': 'Grand County',
        '49021': 'Iron County',
        '49023': 'Juab County',
        '49025': 'Kane County',
        '49027': 'Millard County',
        '49029': 'Morgan County',
        '49031': 'Piute County',
        '49033': 'Rich County',
        '49035': 'Salt Lake County',
        '49037': 'San Juan County',
        '49039': 'Sanpete County',
        '49041': 'Sevier County',
        '49043': 'Summit County',
        '49045': 'Tooele County',
        '49047': 'Utah County',
        '49049': 'Uintah County',
        '49051': 'Wasatch County',
        '49053': 'Washington County',
        '49055': 'Wayne County',
        '49057': 'Weber County'
    }
    
    census['COUNTY'] = census['COUNTYFP'].map(utah_counties)
    
    # Rename columns to match GerryChain conventions
    census = census.rename(columns={
        'P0010001': 'TOT_POP',
        'P0040001': 'VAP',  # Voting Age Population
        'P0010002': 'NOT_HISP',
        'P0010003': 'HISP_POP',
        'P0010004': 'WHITE_POP',
        'P0010005': 'BLACK_POP',
        'P0010006': 'NATIVE_POP',
        'P0010007': 'ASIAN_POP',
        'P0010008': 'PACIFIC_POP',
        'P0010009': 'OTHER_POP',
        'P0010010': 'TWO_OR_MORE_POP'
    })
    
    print(f"Loaded census data: {len(census)} blocks")
    return census

def load_election_data():
    """Load election data from all years."""
    print("Loading election data...")
    
    election_data = {}
    
    # 2016
    gdf_2016 = gpd.read_file("data/disagg/ut_2016_gen_2020_blocks/ut_2016_gen_2020_blocks.shp")
    election_data[2016] = gdf_2016[['GEOID20'] + [col for col in gdf_2016.columns if col.startswith('G16')]]
    
    # 2018
    gdf_2018 = gpd.read_file("data/disagg/ut_2018_gen_2020_blocks/ut_2018_gen_2020_blocks.shp")
    election_data[2018] = gdf_2018[['GEOID20'] + [col for col in gdf_2018.columns if col.startswith('G18')]]
    
    # 2020
    gdf_2020 = gpd.read_file("data/disagg/ut_2020_gen_2020_blocks/ut_2020_gen_2020_blocks.shp")
    election_data[2020] = gdf_2020[['GEOID20'] + [col for col in gdf_2020.columns if col.startswith('G20')]]
    
    # 2024
    gdf_2024 = gpd.read_file("data/disagg/ut_2024_gen_2020_blocks/ut_2024_gen_2020_blocks.shp")
    election_data[2024] = gdf_2024[['GEOID20'] + [col for col in gdf_2024.columns if col.startswith('G24')]]
    
    print(f"Loaded election data for years: {list(election_data.keys())}")
    return election_data

def load_congressional_districts():
    """Load HB2004 congressional districts."""
    print("Loading HB2004 congressional districts...")
    
    hb2004 = gpd.read_file("plans/CONG/HB2004/HB2004.shp")
    
    # Rename for clarity
    hb2004 = hb2004.rename(columns={'DISTRICTNO': 'CONGDIST'})
    
    print(f"Loaded {len(hb2004)} congressional districts")
    return hb2004

def load_state_legislative_districts():
    """Load state house and senate district data."""
    print("Loading state legislative districts...")
    
    # Load state house districts
    sldl = gpd.read_file("plans/SLDL/ut_sldl_2021/ut_sldl_2021.shp")
    sldl = sldl[['DISTRICTNO', 'geometry']].rename(columns={'DISTRICTNO': 'UTHOUDIST'})
    
    # Load state senate districts
    sldu = gpd.read_file("plans/SLDU/ut_sldu_2021/ut_sldu_2021.shp")
    sldu = sldu[['DISTRICTNO', 'geometry']].rename(columns={'DISTRICTNO': 'UTSENDIST'})
    
    print(f"Loaded {len(sldl)} state house districts and {len(sldu)} state senate districts")
    return sldl, sldu

def load_city_data():
    """Load city data from blocks_whichCity.csv."""
    print("Loading city data from blocks_whichCity.csv...")
    
    # Load the CSV file
    city_data = pd.read_csv("data/raw/blocks_whichCity.csv")
    
    # Create municipality column
    # For unincorporated areas (missing PlaceName), use GEOID20
    # For incorporated areas, use PlaceName
    city_data['MUNI'] = city_data.apply(
        lambda row: row['GEOID20'] if pd.isna(row['PlaceName']) else row['PlaceName'], 
        axis=1
    )
    
    # Keep only necessary columns
    city_data = city_data[['GEOID20', 'MUNI']].copy()
    
    print(f"Loaded municipality data for {len(city_data)} blocks")
    print(f"  - Incorporated areas: {len(city_data[~city_data['MUNI'].str.match(r'^\d+$', na=False)])}")
    print(f"  - Unincorporated areas: {len(city_data[city_data['MUNI'].str.match(r'^\d+$', na=False)])}")
    return city_data

def load_precinct_data():
    """Load precinct data for precinct-level aggregation."""
    print("Loading precinct data...")
    
    # Load all precinct data
    precincts = gpd.read_file("data/raw/ut_2024_gen_prec/ut_2024_gen_all_prec/ut_2024_gen_all_prec.shp")
    
    # Keep only necessary columns
    precinct_data = precincts[['UNIQUE_ID', 'COUNTYFP', 'COUNTY', 'geometry']].copy()
    
    print(f"Loaded {len(precinct_data)} precincts")
    return precinct_data

def process_election_results(election_data):
    """Process election results to extract Republican, Democratic, and Other votes."""
    print("Processing election results...")
    
    # Define office mappings for each year (using GerryChain naming convention)
    office_mappings = {
        2016: {
            'PRES': ['G16PRERTRU', 'G16PREDCLI'],  # Trump, Clinton
            'GOV': ['G16GOVRHER', 'G16GOVDWEI'],   # Herbert, Weinholtz
            'ATG': ['G16ATGRREY', 'G16ATGDHAR'],  # Reyes, Harrison
            'AUD': ['G16AUDRDOU', 'G16AUDDMIT'],    # Dougall, Mitchell
            'TRE': ['G16TRERDAM', 'G16TREDHAN']   # Damschen, Hansen
        },
        2018: {
            'USS': ['G18USSRROM', 'G18USSDWIL']   # Romney, Wilson
        },
        2020: {
            'PRES': ['G20PRERTRU', 'G20PREDBID'],  # Trump, Biden
            'GOV': ['G20GOVRCOX', 'G20GOVDPET'],   # Cox, Peterson
            'ATG': ['G20ATGRREY', 'G20ATGDSKO'],  # Reyes, Skordas
            'AUD': ['G20AUDRDOU', 'G20AUDCOST'],    # Dougall, Costello
            'TRE': ['G20TRERDAM', 'G20TRELSPE']   # Damschen, Spendlove
        },
        2024: {
            'PRES': ['G24PRERTRU', 'G24PREDHAR'],  # Trump, Harris
            'USS': ['G24USSRCUR', 'G24USSDGLE'],  # Curtis, Gleich
            'GOV': ['G24GOVRHEN', 'G24GOVDCUM'],   # Henderson, Cummings
            'ATG': ['G24ATGRBRO', 'G24ATGDBAU'],  # Brown, Bauer
            'AUD': ['G24AUDRCAN', 'G24AUDDVOU'],    # Randall, Vought
            'TRE': ['G24TREROAK', 'G24TREDHAN']   # Oaks, Hansen
        }
    }
    
    processed_results = {}
    
    for year, data in election_data.items():
        print(f"Processing {year} election data...")
        
        if year not in office_mappings:
            continue
            
        year_results = data[['GEOID20']].copy()
        
        for office, (rep_col, dem_col) in office_mappings[year].items():
            if rep_col in data.columns and dem_col in data.columns:
                # Get Republican votes (using GerryChain naming: office + year + R)
                year_results[f'{office}{str(year)[2:]}R'] = data[rep_col].fillna(0)
                
                # Get Democratic votes (using GerryChain naming: office + year + D)
                year_results[f'{office}{str(year)[2:]}D'] = data[dem_col].fillna(0)
                
                # Calculate Other votes (total votes - rep - dem)
                # First, find all vote columns for this office
                office_cols = [col for col in data.columns if col.startswith(f'G{str(year)[2:]}') and 
                              any(office_code in col for office_code in ['PRE', 'GOV', 'USS', 'ATG', 'AUD', 'TRE'])]
                
                # Calculate total votes for this office
                total_votes = data[office_cols].sum(axis=1)
                rep_votes = data[rep_col].fillna(0)
                dem_votes = data[dem_col].fillna(0)
                
                # Other votes (using GerryChain naming: office + year + O)
                year_results[f'{office}{str(year)[2:]}O'] = (total_votes - rep_votes - dem_votes).clip(lower=0)
        
        processed_results[year] = year_results
    
    return processed_results

def merge_unincorporated_zero_pop_blocks(blocks):
    """
    Merge adjacent unincorporated blocks with zero population that are identical in all attributes.
    This reduces file size and makes the data more manageable for GerryChain analysis.
    Only merges blocks that have identical district assignments, county, and other attributes.
    """
    print("Merging adjacent unincorporated zero-population blocks with identical attributes...")
    
    # Identify unincorporated zero-population blocks
    uninc_zero_pop = blocks[
        (blocks['TOT_POP'] == 0) & 
        (blocks['MUNI'].str.match(r'^\d+$', na=False))  # Unincorporated (numeric MUNI)
    ].copy()
    
    print(f"Found {len(uninc_zero_pop)} unincorporated zero-population blocks")
    
    if len(uninc_zero_pop) == 0:
        print("No unincorporated zero-population blocks to merge")
        return blocks
    
    # Define the attributes that must be identical for blocks to be merged
    # These are the non-geometric, non-numeric attributes that should be the same
    attribute_cols = [
        'COUNTYFP', 'COUNTY', 'STATEFP', 'CONGDIST', 'UTHOUDIST', 'UTSENDIST', 'PRECINCT_ID'
    ]
    
    # Filter to only include columns that exist in the dataframe
    attribute_cols = [col for col in attribute_cols if col in uninc_zero_pop.columns]
    
    print(f"Grouping by attributes: {attribute_cols}")
    
    # Group blocks by identical attributes
    grouped = uninc_zero_pop.groupby(attribute_cols)
    
    merged_blocks = []
    total_original_blocks = 0
    total_merged_blocks = 0
    
    for group_key, group_blocks in grouped:
        total_original_blocks += len(group_blocks)
        
        if len(group_blocks) == 1:
            # Single block, keep as is but update identifier
            merged_block = group_blocks.iloc[0].copy()
            merged_block['GEOID20'] = f"MERGED_{group_key[0]}_{len(group_blocks)}"  # Use first attribute as identifier
            merged_block['MUNI'] = f"MERGED_{group_key[0]}_{len(group_blocks)}"
            merged_blocks.append(merged_block)
            total_merged_blocks += 1
        else:
            # Merge multiple blocks with identical attributes
            print(f"  Merging {len(group_blocks)} blocks with attributes: {dict(zip(attribute_cols, group_key))}")
            
            # Create merged geometry
            merged_geom = unary_union(group_blocks.geometry.tolist())
            
            # Create merged block with combined attributes
            merged_block = group_blocks.iloc[0].copy()
            merged_block['geometry'] = merged_geom
            merged_block['GEOID20'] = f"MERGED_{group_key[0]}_{len(group_blocks)}"
            merged_block['MUNI'] = f"MERGED_{group_key[0]}_{len(group_blocks)}"
            
            # Sum all numeric columns (population, vote counts, etc.)
            numeric_cols = group_blocks.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in attribute_cols:  # Don't sum the grouping columns
                    merged_block[col] = group_blocks[col].sum()
            
            merged_blocks.append(merged_block)
            total_merged_blocks += 1
    
    # Convert merged blocks to GeoDataFrame
    if merged_blocks:
        merged_gdf = gpd.GeoDataFrame(merged_blocks, crs=blocks.crs)
        
        # Combine with non-merged blocks
        non_merged = blocks[
            ~((blocks['TOT_POP'] == 0) & 
              (blocks['MUNI'].str.match(r'^\d+$', na=False)))
        ]
        
        result = pd.concat([non_merged, merged_gdf], ignore_index=True)
        
        print(f"Merged {total_original_blocks} blocks into {total_merged_blocks} merged blocks")
        print(f"Total blocks after merging: {len(result)} (reduced by {total_original_blocks - total_merged_blocks})")
        
        return result
    else:
        print("No blocks were merged")
        return blocks

def assign_districts(blocks, cong, sldl, sldu, city_data, precinct_data):
    """Assign congressional and state legislative districts to blocks using spatial join."""
    print("Assigning districts to blocks...")
    
    # Ensure all are in the same CRS
    if blocks.crs != cong.crs:
        blocks = blocks.to_crs(cong.crs)
    if sldl.crs != cong.crs:
        sldl = sldl.to_crs(cong.crs)
    if sldu.crs != cong.crs:
        sldu = sldu.to_crs(cong.crs)
    if precinct_data.crs != cong.crs:
        precinct_data = precinct_data.to_crs(cong.crs)
    
    # Reset index to avoid conflicts with spatial joins
    blocks = blocks.reset_index(drop=True)
    cong = cong.reset_index(drop=True)
    sldl = sldl.reset_index(drop=True)
    sldu = sldu.reset_index(drop=True)
    
    # Assign congressional districts
    print("Assigning congressional districts...")
    blocks_with_districts = gpd.sjoin(blocks, cong[['CONGDIST', 'geometry']], 
                                     how='left', predicate='intersects')
    
    # Handle any blocks that didn't get assigned to congressional districts
    unassigned_cong = blocks_with_districts['CONGDIST'].isna()
    if unassigned_cong.sum() > 0:
        print(f"Warning: {unassigned_cong.sum()} blocks were not assigned to a congressional district")
        # For unassigned blocks, assign to the nearest district
        unassigned_blocks = blocks_with_districts[unassigned_cong]
        for idx, block in unassigned_blocks.iterrows():
            distances = cong.geometry.distance(block.geometry)
            nearest_district = cong.iloc[distances.idxmin()]['CONGDIST']
            blocks_with_districts.loc[idx, 'CONGDIST'] = nearest_district
    
    # Drop index_right column if it exists and reset index before next join
    if 'index_right' in blocks_with_districts.columns:
        blocks_with_districts = blocks_with_districts.drop(columns=['index_right'])
    blocks_with_districts = blocks_with_districts.reset_index(drop=True)
    
    # Assign state house districts
    print("Assigning state house districts...")
    blocks_with_districts = gpd.sjoin(blocks_with_districts, sldl[['UTHOUDIST', 'geometry']], 
                                     how='left', predicate='intersects')
    
    # Handle any blocks that didn't get assigned to state house districts
    unassigned_house = blocks_with_districts['UTHOUDIST'].isna()
    if unassigned_house.sum() > 0:
        print(f"Warning: {unassigned_house.sum()} blocks were not assigned to a state house district")
        # For unassigned blocks, assign to the nearest district
        unassigned_blocks = blocks_with_districts[unassigned_house]
        for idx, block in unassigned_blocks.iterrows():
            distances = sldl.geometry.distance(block.geometry)
            nearest_district = sldl.iloc[distances.idxmin()]['UTHOUDIST']
            blocks_with_districts.loc[idx, 'UTHOUDIST'] = nearest_district
    
    # Drop index_right column if it exists and reset index before next join
    if 'index_right' in blocks_with_districts.columns:
        blocks_with_districts = blocks_with_districts.drop(columns=['index_right'])
    blocks_with_districts = blocks_with_districts.reset_index(drop=True)
    
    # Assign state senate districts
    print("Assigning state senate districts...")
    blocks_with_districts = gpd.sjoin(blocks_with_districts, sldu[['UTSENDIST', 'geometry']], 
                                     how='left', predicate='intersects')
    
    # Handle any blocks that didn't get assigned to state senate districts
    unassigned_senate = blocks_with_districts['UTSENDIST'].isna()
    if unassigned_senate.sum() > 0:
        print(f"Warning: {unassigned_senate.sum()} blocks were not assigned to a state senate district")
        # For unassigned blocks, assign to the nearest district
        unassigned_blocks = blocks_with_districts[unassigned_senate]
        for idx, block in unassigned_blocks.iterrows():
            distances = sldu.geometry.distance(block.geometry)
            nearest_district = sldu.iloc[distances.idxmin()]['UTSENDIST']
            blocks_with_districts.loc[idx, 'UTSENDIST'] = nearest_district
    
    # Drop index_right column if it exists and reset index before precinct join
    if 'index_right' in blocks_with_districts.columns:
        blocks_with_districts = blocks_with_districts.drop(columns=['index_right'])
    blocks_with_districts = blocks_with_districts.reset_index(drop=True)
    precinct_data = precinct_data.reset_index(drop=True)
    
    # Assign precinct information
    print("Assigning precinct information...")
    blocks_with_districts = gpd.sjoin(blocks_with_districts, precinct_data[['UNIQUE_ID', 'geometry']], 
                                     how='left', predicate='intersects')
    
    # Handle any blocks that didn't get assigned to a precinct
    unassigned_precinct = blocks_with_districts['UNIQUE_ID'].isna()
    if unassigned_precinct.sum() > 0:
        print(f"Warning: {unassigned_precinct.sum()} blocks were not assigned to a precinct")
        # For unassigned blocks, assign to the nearest precinct
        unassigned_blocks = blocks_with_districts[unassigned_precinct]
        for idx, block in unassigned_blocks.iterrows():
            distances = precinct_data.geometry.distance(block.geometry)
            nearest_precinct = precinct_data.iloc[distances.idxmin()]['UNIQUE_ID']
            blocks_with_districts.loc[idx, 'UNIQUE_ID'] = nearest_precinct
    
    # Rename UNIQUE_ID to PRECINCT_ID for clarity
    blocks_with_districts = blocks_with_districts.rename(columns={'UNIQUE_ID': 'PRECINCT_ID'})
    
    # Assign municipality information from CSV data
    print("Assigning municipality information from CSV data...")
    # Ensure GEOID20 columns are the same type
    blocks_with_districts['GEOID20'] = blocks_with_districts['GEOID20'].astype(str)
    city_data['GEOID20'] = city_data['GEOID20'].astype(str)
    blocks_with_districts = blocks_with_districts.merge(city_data, on='GEOID20', how='left')
    
    # Handle any blocks that didn't get assigned to a municipality
    unassigned_muni = blocks_with_districts['MUNI'].isna()
    if unassigned_muni.sum() > 0:
        print(f"Warning: {unassigned_muni.sum()} blocks were not assigned to a municipality")
        # For unassigned blocks, use GEOID20 as fallback
        blocks_with_districts.loc[unassigned_muni, 'MUNI'] = blocks_with_districts.loc[unassigned_muni, 'GEOID20']
    
    return blocks_with_districts

def create_block_level_geojson():
    """Create block-level GeoJSON file."""
    print("Creating block-level GeoJSON...")
    
    # Load all data
    census = load_census_data()
    election_data = load_election_data()
    hb2004 = load_congressional_districts()
    sldl, sldu = load_state_legislative_districts()
    city_data = load_city_data()
    
    # Process election results
    processed_elections = process_election_results(election_data)
    
    # Start with census data
    result = census.copy()
    
    # Add election results for each year
    for year, year_data in processed_elections.items():
        print(f"Adding {year} election data...")
        result = result.merge(year_data, on='GEOID20', how='left')
    
    # Load precinct data for precinct assignment
    precinct_data = load_precinct_data()
    
    # Assign all districts (congressional, state house, state senate), precinct, and municipality
    result = assign_districts(result, hb2004, sldl, sldu, city_data, precinct_data)
    
    # Merge adjacent unincorporated zero-population blocks to reduce file size
    result = merge_unincorporated_zero_pop_blocks(result)
    
    # Add boundary_node field (False for all blocks)
    result['boundary_node'] = False
    
    # Fill NaN values with 0 for vote columns
    vote_cols = [col for col in result.columns if any(suffix in col for suffix in ['R', 'D', 'O']) and col.endswith(('R', 'D', 'O'))]
    result[vote_cols] = result[vote_cols].fillna(0)
    
    # Convert to GeoJSON and save
    output_path = "data/UT_blocks.geojson"
    result.to_file(output_path, driver='GeoJSON')
    
    print(f"Block-level GeoJSON saved to: {output_path}")
    print(f"Total blocks: {len(result)}")
    print(f"Columns: {len(result.columns)}")
    
    return result

def create_precinct_level_geojson(block_data):
    """Create precinct-level GeoJSON by aggregating block data."""
    print("Creating precinct-level GeoJSON...")
    
    # Load precinct data to get precinct boundaries
    precinct_data = load_precinct_data()
    
    # We need to assign blocks to precincts first
    # This is a spatial join operation
    print("Assigning blocks to precincts...")
    
    # Ensure both are in the same CRS
    if block_data.crs != precinct_data.crs:
        block_data = block_data.to_crs(precinct_data.crs)
    
    # Drop index_right column if it exists and reset index to avoid conflicts with spatial joins
    if 'index_right' in block_data.columns:
        block_data = block_data.drop(columns=['index_right'])
    block_data = block_data.reset_index(drop=True)
    precinct_data = precinct_data.reset_index(drop=True)
    
    # Assign blocks to precincts using spatial join
    blocks_with_precincts = gpd.sjoin(block_data, precinct_data[['UNIQUE_ID', 'geometry']], 
                                     how='left', predicate='intersects')
    
    # Handle any blocks that didn't get assigned to a precinct
    unassigned_precinct = blocks_with_precincts['UNIQUE_ID'].isna()
    if unassigned_precinct.sum() > 0:
        print(f"Warning: {unassigned_precinct.sum()} blocks were not assigned to a precinct")
        # For unassigned blocks, assign to the nearest precinct
        unassigned_blocks = blocks_with_precincts[unassigned_precinct]
        for idx, block in unassigned_blocks.iterrows():
            distances = precinct_data.geometry.distance(block.geometry)
            nearest_precinct = precinct_data.iloc[distances.idxmin()]['UNIQUE_ID']
            blocks_with_precincts.loc[idx, 'UNIQUE_ID'] = nearest_precinct
    
    # Aggregate block data to precinct level
    print("Aggregating block data to precincts...")
    
    # Get numeric columns for aggregation (exclude geometry, text, and ID columns)
    numeric_cols = blocks_with_precincts.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [col for col in numeric_cols if col not in ['GEOID20', 'UNIQUE_ID']]
    
    # Group by precinct and sum numeric columns
    precinct_agg = blocks_with_precincts.groupby('UNIQUE_ID')[agg_cols].sum().reset_index()
    
    # For non-numeric columns, take the first value (they should be the same within each precinct)
    text_cols = ['GEOID20', 'MUNI', 'COUNTY', 'CONGDIST', 'UTHOUDIST', 'UTSENDIST', 'STATEFP', 'COUNTYFP', 'PRECINCT_ID']
    text_cols = [col for col in text_cols if col in blocks_with_precincts.columns]
    
    if text_cols:
        text_agg = blocks_with_precincts.groupby('UNIQUE_ID')[text_cols].first().reset_index()
        precinct_agg = precinct_agg.merge(text_agg, on='UNIQUE_ID', how='left')
    
    # Merge with precinct geometries
    precinct_result = precinct_data.merge(precinct_agg, on='UNIQUE_ID', how='left')
    
    # Fill NaN values with 0 for vote columns
    vote_cols = [col for col in precinct_result.columns if any(suffix in col for suffix in ['R', 'D', 'O']) and col.endswith(('R', 'D', 'O'))]
    precinct_result[vote_cols] = precinct_result[vote_cols].fillna(0)
    
    # Convert to GeoJSON and save
    output_path = "data/UT_precincts.geojson"
    precinct_result.to_file(output_path, driver='GeoJSON')
    
    print(f"Precinct-level GeoJSON saved to: {output_path}")
    print(f"Total precincts: {len(precinct_result)}")
    print(f"Columns: {len(precinct_result.columns)}")
    
    return precinct_result

def main():
    """Main function to create both GeoJSON files."""
    print("Creating Utah Election Data GeoJSON Files")
    print("=" * 50)
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs("data", exist_ok=True)
    
    # Create block-level GeoJSON
    block_data = create_block_level_geojson()
    
    # Create precinct-level GeoJSON
    precinct_data = create_precinct_level_geojson(block_data)
    
    print("\n" + "=" * 50)
    print("GeoJSON creation completed successfully!")
    print("\nFiles created:")
    print("1. data/UT_blocks.geojson - Block-level data")
    print("2. data/UT_precincts.geojson - Precinct-level data")
    print("\nData includes:")
    print("- Census demographics (population, race/ethnicity)")
    print("- Election results for President, Governor, US Senate, Attorney General, Auditor, Treasurer")
    print("- Republican, Democratic, and Other party vote totals")
    print("- HB2004 congressional district assignments")
    print("- State House and State Senate district assignments")
    print("- County and municipality information")

if __name__ == "__main__":
    main()
