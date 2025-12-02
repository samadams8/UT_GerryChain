from statistics import mean, median, stdev
import pandas as pd
from gerrychain.partition import Partition
from typing import List, Tuple

def get_num_split_munis(partition):
    try:
        muni_ls = partition["muni_locality_splits"]
        return int(muni_ls.get("num_split_localities", 0))
    except Exception:
        return 0

def get_num_split_counties(partition):
    try:
        county_ls = partition["county_locality_splits"]
        return int(county_ls.get("num_split_localities", 0))
    except Exception:
        return 0

def aggregate_partisan_metrics(df: pd.DataFrame, aggregation_method: str = 'mean') -> dict:
    """
    Computes aggregated vote shares and partisan indices for statewide and
    district-level results from a long-format DataFrame.

    Metrics:
    - Vote Share: Proportion of votes for a party (D/R) relative to total votes ('-').
    - Partisan Index: Fraction of two-party votes (D+R) won by a party.

    Aggregation:
    - Metrics are calculated for each (District, Election).
    - "Statewide" ('S') is derived by summing all districts *per election*
      before calculating metrics.
    - Metrics are then aggregated across all elections for each district (and 'S')
      using the specified method ('mean' or 'median').

    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns ['District', 'Election', 'Party', 'VoteCount'].
    - aggregation_method (str): The method to use for aggregation ('mean' or 'median').
                                Defaults to 'mean'.

    Returns:
    - dict: A dictionary where keys are district IDs (including 0) or 'S'
            for statewide, and values are dictionaries containing the
            aggregated metrics:
            {'VoteShare_D', 'VoteShare_R', 'PartisanIndex_D', 'PartisanIndex_R'}.
    """
    
    # --- 1. Validation ---
    if aggregation_method not in ['mean', 'median']:
        raise ValueError("aggregation_method must be 'mean' or 'median'")

    # --- 2. Create Statewide Data ---
    # Group by Election and Party, summing VoteCount across *all* districts
    # to get statewide totals for each party in each election.
    statewide_df = df.groupby(['Election', 'Party'])['VoteCount'].sum().reset_index()
    # Assign 'S' as the District
    statewide_df['District'] = 'S'
    
    # Create a copy to avoid SettingWithCopyWarning and ensure type safety
    df_districts = df.copy()
    df_districts['District'] = df_districts['District'].astype(str)

    # Combine original district data with new statewide data
    combined_df = pd.concat([df_districts, statewide_df], ignore_index=True)

    # --- 3. Pivot Data ---
    # Pivot the *combined* data. Now 'S' is treated as a separate "district".
    try:
        df_wide = combined_df.pivot_table(
            index=['District', 'Election'],
            columns='Party',
            values='VoteCount',
            fill_value=0
        ).reset_index()
    except Exception as e:
        print(f"Error during pivoting. Ensure columns 'District', 'Election', 'Party', 'VoteCount' exist. Error: {e}")
        return {}

    # Clean up column names left by pivot
    df_wide.columns.name = None

    # --- 4. Ensure Columns Exist and Rename ---
    # If a party (e.g., 'D', 'R', or '-') had no votes in the *entire*
    # dataset, the pivot table won't create the column. We add it as 0.
    for col in ['D', 'R', '-']:
        if col not in df_wide.columns:
            df_wide[col] = 0
            
    # Rename '-' to 'TotalVotes' for clarity
    df_wide = df_wide.rename(columns={'-': 'TotalVotes'})

    # --- 5. Calculate Per-Election Metrics ---

    # Get a map of {Election -> StatewideTotalVotes} to calculate district proportion
    # We get this from our 'S' district rows
    statewide_totals_map = df_wide[df_wide['District'] == 'S'].set_index('Election')['TotalVotes']
    # Map these totals to a new column; every row for a given election gets the same value
    df_wide['StatewideTotalVotes'] = df_wide['Election'].map(statewide_totals_map)
    
    # A. Calculate Vote Share (vs. Total Votes)
    # VoteShare_D = D_Votes / TotalVotes
    df_wide['VoteShare_D'] = df_wide['D'] / df_wide['TotalVotes']
    df_wide['VoteShare_R'] = df_wide['R'] / df_wide['TotalVotes']

    # B. Calculate Partisan Index (Two-Party Vote Share)
    # PartisanIndex_D = D_Votes / (D_Votes + R_Votes)
    two_party_votes = df_wide['D'] + df_wide['R']
    df_wide['PartisanIndex_D'] = df_wide['D'] / two_party_votes
    df_wide['PartisanIndex_R'] = df_wide['R'] / two_party_votes

    # C. Calculate Proportion of Statewide Vote
    # (District Total Votes) / (Statewide Total Votes) for each election
    # For the 'S' district, this will correctly be 1.0
    df_wide['StatewideVoteProportion'] = df_wide['TotalVotes'] / df_wide['StatewideTotalVotes']

    # --- 6. Aggregate Metrics ---
    
    # Define the metrics we want to aggregate
    metrics_to_agg = [
        'VoteShare_D', 'VoteShare_R',
        'PartisanIndex_D', 'PartisanIndex_R',
        'StatewideVoteProportion'
    ]
    
    # Group by District (which now includes 'S'), select our new metrics,
    # and aggregate using the specified method (mean or median).
    aggregated_df = df_wide.groupby('District')[metrics_to_agg].agg(
        aggregation_method
    ).reset_index()

    # --- 7. Format Output as Dictionary ---
    
    # Set District as index for to_dict
    aggregated_df = aggregated_df.set_index('District')

    # Convert to the desired dictionary format
    output_dict = aggregated_df.to_dict(orient='index')

    return output_dict

def _parties(pmetrics: dict) -> List[str]:
    """Returns the parties sorted by decreasing statewide vote share"""
    parties = [p[-1] for p in filter(lambda x: x.startswith('VoteShare_'), pmetrics['S'].keys())]
    parties.sort(key=lambda x: pmetrics['S'][f'VoteShare_{x}'], reverse=True)
    return parties

def efficiency_gap(pmetrics: dict) -> float:
    """
    Computes the efficiency gap, which is the difference in the proportion of wasted votes between the majority and minority parties.
    
    A negative efficiency gap indicates that the majority party is more efficient at converting votes into seats than the minority party, since
      `efficiency_gap = wasted_votes_majority - wasted_votes_minority`
    """
    parties = _parties(pmetrics)
    wasted_votes = {p: 0 for p in parties}
    for district, data in pmetrics.items(): 
        if district == 'S':
            continue
        
        # Use PartisanIndex for all calculations to ensure consistency
        # Standard Efficiency Gap uses 50% of 2-party vote as threshold
        win_threshold = 0.5
        district_turnout = data['StatewideVoteProportion']
        
        for p in parties:
            v = data[f'PartisanIndex_{p}']
            if v > win_threshold:
                # Winner wastes votes above 50%
                wasted_votes[p] += (v - win_threshold) * district_turnout
            else:
                # Loser wastes all votes
                wasted_votes[p] += v * district_turnout

    return wasted_votes[parties[0]] - wasted_votes[parties[1]]

def partisan_bias(pmetrics: dict) -> float:
    mp = _parties(pmetrics)[0]

    party_shares = [
        pmetrics[d][f'PartisanIndex_{mp}']
        for d in filter(lambda d: d != 'S', pmetrics.keys())
    ]
    mean_share = mean(party_shares)
    above_mean_districts = len(list(filter(lambda s: s > mean_share, party_shares)))
    return (above_mean_districts / len(party_shares)) - 0.5

def partisan_bias_utah(pmetrics: dict) -> float:
    # Determine which party is the "majority" party.
    # This will be the reference party for partisan bias calculations.
    mp = _parties(pmetrics)[0]

    # Get statewide partisan index
    state_index = pmetrics['S'][f'PartisanIndex_{mp}']
    # Compute uniform swing to achieve 50% statewide vote share
    swing = 0.5 - state_index

    seats_won, seats_lost = 0, 0
    # Count the number of seats under hypothetical uniform swing
    for district in pmetrics.keys():
        if district == 'S':
            continue
        dist_index_swing = pmetrics[district][f'PartisanIndex_{mp}'] + swing
        if dist_index_swing > 0.5:
            seats_won += 1
        elif dist_index_swing < 0.5:
            seats_lost += 1
        else:
            seats_won += 0.5
            seats_lost += 0.5

    total_seats = seats_won + seats_lost
    even_split_target = total_seats / 2.0

    bias = seats_won - even_split_target
    
    return bias

def mean_median(pmetrics: dict) -> dict[str, float]:
    results = {}
    for party in _parties(pmetrics):
        district_shares = []
        for district in pmetrics.keys():
            if district == 'S':
                continue
            district_shares.append(pmetrics[district][f'VoteShare_{party}'])
        results[party] = mean(district_shares) - median(district_shares)
    return results

def majority_partisan_shares(pmetrics: dict) -> dict[str, float]:
    """Returns the majority party's partisan share in each district"""
    mp = _parties(pmetrics)[0]
    results = {}
    for district in pmetrics.keys():
        if district == 'S':
            continue
        results[district] = pmetrics[district][f'PartisanIndex_{mp}']
    return results

def majority_seats(pmetrics: dict) -> int:
    """Returns the number of districts won by the majority party"""
    mp = _parties(pmetrics)[0]
    seats = 0
    for district in pmetrics.keys():
        if district == 'S':
            continue
        if pmetrics[district][f'PartisanIndex_{mp}'] > 0.5:
            seats += 1
    return seats

def stdev_partisan_share(pmetrics: dict) -> float:
    """Returns the standard deviation of the majority party's partisan shares"""
    mp = _parties(pmetrics)[0]
    partisan_shares = [
        pmetrics[d][f'PartisanIndex_{mp}']
        for d in filter(lambda d: d != 'S', pmetrics.keys())
    ]
    return stdev(partisan_shares)

def tabulate_partisan_data(
    partition: Partition,
    elections: List[str],
    parties: List[str]
) -> pd.DataFrame:
    df = pd.DataFrame(columns=['District','Election','Party','VoteCount'])
    for e in elections:
        u = partition[e]
        counts = {}
        for r in u.regions.keys():
            counts[r] = {}
            for p in parties:
                counts[r][p] = 0
                for ep in u.election.parties:
                    if ep[0] == p:
                        counts[r][p] += u.count(ep, region=r)
            
            for p in parties:
                df.loc[len(df)] = {
                    'District': r,
                    'Election': e,
                    'Party': p,
                    'VoteCount': counts[r][p]
                }
    return df  

def build_locality_name_maps(partition):
    county_id_to_name = {}
    muni_id_to_name = {}
    for node in partition.graph.nodes:
        nd = partition.graph.nodes[node]
        cid = nd.get("COUNTYID")
        cname = nd.get("COUNTYNAME") or nd.get("COUNTY")
        if cid is not None and cid != "" and cid not in county_id_to_name and cname:
            county_id_to_name[cid] = cname
        mid = nd.get("MUNIID")
        mname = nd.get("MUNINAME")
        if mid is not None and mid != "" and mid not in muni_id_to_name and mname:
            muni_id_to_name[mid] = mname
    return county_id_to_name, muni_id_to_name

def compute_split_name_lists(partition):
    county_id_to_name, muni_id_to_name = build_locality_name_maps(partition)
    county_to_districts = {}
    for node in partition.graph.nodes:
        node_data = partition.graph.nodes[node]
        county_id = node_data.get("COUNTYID")
        if county_id:
            dist = partition.assignment[node]
            if county_id not in county_to_districts:
                county_to_districts[county_id] = set()
            county_to_districts[county_id].add(dist)
    split_counties = sorted([cid for cid, dists in county_to_districts.items() if len(dists) > 1])
    split_counties_names = sorted([county_id_to_name.get(cid, str(cid)) for cid in split_counties])

    muni_to_districts = {}
    for node in partition.graph.nodes:
        node_data = partition.graph.nodes[node]
        muni_id = node_data.get("MUNIID")
        if muni_id:
            dist = partition.assignment[node]
            if muni_id not in muni_to_districts:
                muni_to_districts[muni_id] = set()
            muni_to_districts[muni_id].add(dist)
    split_munis = sorted([m for m, dists in muni_to_districts.items() if len(dists) > 1])
    split_munis_names = sorted([muni_id_to_name.get(mid, str(mid)) for mid in split_munis])

    return split_counties_names, split_munis_names
