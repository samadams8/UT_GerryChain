from gerrychain.metrics import partisan_bias, mean_median, efficiency_gap
import numpy as np


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


def calculate_partisan_metrics(partition, available_elections):
    metrics = {}
    if not available_elections:
        return metrics
    for election in available_elections:
        if election in partition.updaters:
            try:
                election_results = partition[election]
                metrics[f"{election}_efficiency_gap"] = efficiency_gap(election_results)
                metrics[f"{election}_mean_median"] = mean_median(election_results)
                metrics[f"{election}_partisan_bias"] = partisan_bias(election_results)
            except Exception as e:
                metrics[f"{election}_efficiency_gap"] = None
                metrics[f"{election}_mean_median"] = None
                metrics[f"{election}_partisan_bias"] = None
    return metrics


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


def compute_split_name_lists(partition, county_id_to_name, muni_id_to_name):
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


