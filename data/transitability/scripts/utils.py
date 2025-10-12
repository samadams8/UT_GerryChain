"""Offline transitability utilities (water-first prune, selective road add-backs).

Pipeline:
- Build base adjacency with shared boundaries
- Prune by buffered water intersection (strict on major waters)
- Add back only pruned edges that are bridged by roads
- Apply hierarchical fallback (same-county) without violating water rule
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import maup
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon


def build_base_edges(precincts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not np.issubdtype(pd.Series(precincts.index).dtype, np.number):
        precincts = precincts.copy()
        precincts.index = np.arange(len(precincts))

    adj_series = maup.adjacencies(precincts, adjacency_type="rook")

    records: List[Dict] = []
    for (u, v), _ in adj_series.items():
        try:
            g_u = precincts.loc[u, "geometry"]
            g_v = precincts.loc[v, "geometry"]
            shared = g_u.boundary.intersection(g_v.boundary)
            blen = float(shared.length)
        except Exception:
            shared = None
            blen = 0.0
        records.append({"u": int(u), "v": int(v), "boundary_len": blen, "geometry": shared})

    return gpd.GeoDataFrame(records, geometry="geometry", crs=precincts.crs)


def _union_buffered_water(lakes: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame, buffer_m: float):
    if lakes.crs != rivers.crs:
        rivers = rivers.to_crs(lakes.crs)
    lb = lakes.copy(); lb["geometry"] = lb.geometry.buffer(buffer_m)
    rb = rivers.copy(); rb["geometry"] = rb.geometry.buffer(buffer_m)
    return gpd.GeoSeries(pd.concat([lb[["geometry"]], rb[["geometry"]]]).unary_union, crs=lakes.crs).iloc[0]


def prune_edges_by_water(
    edges: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    water_threshold: float = 0.05,
    water_buffer_m: int = 150,
    major_columns: Optional[List[str]] = None,
    major_patterns: Optional[List[str]] = None,
) -> gpd.GeoDataFrame:
    if major_columns is None:
        major_columns = ["NAME", "GNIS_NAME"]
    if major_patterns is None:
        major_patterns = [r"Great\s*Salt\s*Lake", r"(Colorado\s*River|Lake\s*Powell)"]

    lakes = lakes.to_crs(edges.crs)
    rivers = rivers.to_crs(edges.crs)
    unioned = _union_buffered_water(lakes, rivers, water_buffer_m)

    def is_major_name(v: Optional[str]) -> bool:
        if not isinstance(v, str):
            return False
        for p in major_patterns:
            if pd.Series([v]).str.contains(p, case=False, regex=True).iloc[0]:
                return True
        return False

    def row_is_major(r: pd.Series) -> bool:
        for c in major_columns:
            if is_major_name(r.get(c)):
                return True
        return False

    lakes_major = lakes[lakes.apply(row_is_major, axis=1)][["geometry"]]
    rivers_major = rivers[rivers.apply(row_is_major, axis=1)][["geometry"]]
    major_union = None
    geoms = []
    if not lakes_major.empty:
        geoms.append(lakes_major.unary_union)
    if not rivers_major.empty:
        geoms.append(rivers_major.unary_union)
    if geoms:
        major_union = unary_union(geoms)

    edges = edges.copy()
    edges["intersects_water"] = False
    edges["intersection_len"] = 0.0
    edges["intersection_ratio"] = 0.0
    edges["removed_by_water"] = False

    mask = edges.geometry.notnull() & edges.intersects(unioned)
    for idx, r in edges[mask].iterrows():
        boundary = r.geometry
        inter = boundary.intersection(unioned)
        inter_len = float(inter.length) if not inter.is_empty else 0.0
        ratio = (inter_len / r.boundary_len) if r.boundary_len > 0 else 0.0
        strict = False
        if inter_len > 0 and major_union is not None and boundary.intersects(major_union):
            strict = True
        removed = strict or (ratio >= float(water_threshold))

        edges.at[idx, "intersects_water"] = inter_len > 0
        edges.at[idx, "intersection_len"] = inter_len
        edges.at[idx, "intersection_ratio"] = ratio
        edges.at[idx, "removed_by_water"] = removed

    return edges


def add_back_edges_via_roads(edges: gpd.GeoDataFrame, precincts: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, road_boundary_buffer_m: int = 15) -> gpd.GeoDataFrame:
    if roads.crs != edges.crs:
        roads = roads.to_crs(edges.crs)
    if precincts.crs != edges.crs:
        precincts = precincts.to_crs(edges.crs)

    edges = edges.copy()
    if "added_by_road" not in edges.columns:
        edges["added_by_road"] = False

    sindex = roads.sindex
    candidates = edges[(edges["removed_by_water"] == True) & edges.geometry.notnull()]

    for idx, r in candidates.iterrows():
        boundary = r.geometry
        corridor = boundary.buffer(road_boundary_buffer_m)
        hits = list(sindex.intersection(corridor.bounds))
        if not hits:
            continue
        roads_near = roads.iloc[hits]
        try:
            u_buf = precincts.loc[r.u, "geometry"].buffer(road_boundary_buffer_m)
            v_buf = precincts.loc[r.v, "geometry"].buffer(road_boundary_buffer_m)
        except Exception:
            continue

        bridged = False
        for _, rr in roads_near.iterrows():
            g = rr.geometry
            if g is None or g.is_empty:
                continue
            if not g.intersects(corridor):
                continue
            if g.intersects(u_buf) and g.intersects(v_buf):
                bridged = True
                break
        if bridged:
            edges.at[idx, "added_by_road"] = True
            edges.at[idx, "removed_by_water"] = False

    return edges


def build_graph_from_edges(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(int(r["u"]), int(r["v"]))
    return G


def apply_hierarchical_fallback_offline(precincts: gpd.GeoDataFrame, edges_base: gpd.GeoDataFrame, edges_after: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    current = edges_after[edges_after["removed_by_water"] == False][["u", "v"]]
    G = build_graph_from_edges(current)
    for n in precincts.index:
        if n not in G:
            G.add_node(int(n))

    orphaned = [n for n, d in G.degree() if d == 0]
    if not orphaned:
        return edges_after

    allowed_pairs = set((min(int(r.u), int(r.v)), max(int(r.u), int(r.v))) for _, r in edges_after.iterrows() if r["removed_by_water"] == False)

    added: List[Dict] = []
    for n in orphaned:
        county_n = precincts.loc[n, "COUNTYID"] if "COUNTYID" in precincts.columns else None
        neigh = edges_base[(edges_base.u == n) | (edges_base.v == n)]
        for _, r in neigh.iterrows():
            u, v = int(r.u), int(r.v)
            other = v if u == n else u
            county_o = precincts.loc[other, "COUNTYID"] if "COUNTYID" in precincts.columns else None
            pair = (min(u, v), max(u, v))
            if county_n and county_o and county_n == county_o and pair in allowed_pairs:
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
                rec = r.to_dict()
                rec.update({"intersects_water": False, "intersection_len": 0.0, "intersection_ratio": 0.0, "removed_by_water": False, "added_by_road": rec.get("added_by_road", False), "fallback_type": "county_fallback"})
                added.append(rec)
                break

    if added:
        added_df = gpd.GeoDataFrame(added, geometry="geometry", crs=edges_after.crs)
        return pd.concat([edges_after, added_df], ignore_index=True)
    return edges_after


def export_artifacts(edges_base: gpd.GeoDataFrame, edges_final: gpd.GeoDataFrame, precincts: gpd.GeoDataFrame, out_dir: Path, export_formats: Optional[List[str]] = None) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if export_formats is None:
        export_formats = ["parquet", "graphml", "json"]

    artifacts: Dict[str, str] = {}
    base_path = out_dir / "edges_base.parquet"
    final_path = out_dir / "edges_final.parquet"
    edges_base.drop(columns=["geometry"], errors="ignore").to_parquet(base_path)
    edges_final.drop(columns=["geometry"], errors="ignore").to_parquet(final_path)
    artifacts["edges_base_parquet"] = str(base_path)
    artifacts["edges_final_parquet"] = str(final_path)

    nodes = pd.DataFrame({"node_id": precincts.index})
    nodes_path = out_dir / "transitability_graph_nodes.parquet"
    nodes.to_parquet(nodes_path)
    artifacts["nodes_parquet"] = str(nodes_path)

    edges_simple = edges_final[["u", "v"]].copy()
    edges_simple_path = out_dir / "transitability_graph_edges.parquet"
    edges_simple.to_parquet(edges_simple_path)
    artifacts["edges_parquet"] = str(edges_simple_path)

    G = build_graph_from_edges(edges_simple)
    if "graphml" in export_formats:
        gml = out_dir / "transitability.graphml"
        nx.write_graphml(G, gml)
        artifacts["graphml"] = str(gml)
    if "json" in export_formats:
        import json
        js = out_dir / "transitability.json"
        with open(js, "w") as f:
            json.dump([{ "source": int(u), "target": int(v)} for u, v in G.edges()], f)
        artifacts["json"] = str(js)

    return artifacts


# --- New: Step 1 road-first pruning (original method) ---

def compute_precincts_with_road_access(precincts: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, buffer_m: float = 500.0) -> pd.Series:
    if roads.crs != precincts.crs:
        roads = roads.to_crs(precincts.crs)
    pbuf = precincts.geometry.buffer(buffer_m)
    # Efficient intersections via maup; for linework, use intersections directly
    inter = maup.intersections(gpd.GeoDataFrame(geometry=pbuf, crs=precincts.crs), roads)
    # Any overlap counts as access
    has_access = pd.Series(False, index=precincts.index)
    if len(inter) > 0:
        has_access.loc[inter.index.get_level_values(0).unique()] = True
    return has_access


def prune_edges_by_road_first(edges_base: gpd.GeoDataFrame, precincts: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, buffer_m: float = 500.0) -> gpd.GeoDataFrame:
    """Original road-based pruning: keep edges only when both precincts have road access.

    Adds columns: has_road_u, has_road_v, kept_by_road (bool), removed_by_road (bool)
    """
    access = compute_precincts_with_road_access(precincts, roads, buffer_m)
    edges = edges_base.copy()
    edges["has_road_u"] = edges["u"].map(access)
    edges["has_road_v"] = edges["v"].map(access)
    edges["kept_by_road"] = edges["has_road_u"] & edges["has_road_v"]
    edges["removed_by_road"] = ~edges["kept_by_road"]
    return edges


# --- New: Step 2 water-mostly pruning with direct road exception ---

def _boundary_water_ratio(boundary: Polygon, unioned_water) -> float:
    if boundary is None or boundary.is_empty:
        return 0.0
    inter = boundary.intersection(unioned_water)
    if inter.is_empty:
        return 0.0
    b_len = float(boundary.length)
    return float(inter.length) / b_len if b_len > 0 else 0.0


def prune_water_mostly_with_road_exception(
    edges_after_road: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    precincts: gpd.GeoDataFrame,
    water_threshold: float = 0.5,
    water_buffer_m: int = 150,
    road_boundary_buffer_m: int = 15,
    major_columns: Optional[List[str]] = None,
    major_patterns: Optional[List[str]] = None,
) -> gpd.GeoDataFrame:
    """Mark edges for removal when most of the shared boundary is water, unless a road bridges it.

    Uses buffered water union and the same road-bridging heuristic as add_back_edges_via_roads.
    Retains and updates columns from `edges_after_road`. Adds/updates:
      - water_ratio
      - removed_by_water (may set True)
      - added_by_road (if exception applies)
    """
    # Prepare water union
    lakes = lakes.to_crs(edges_after_road.crs)
    rivers = rivers.to_crs(edges_after_road.crs)
    unioned = _union_buffered_water(lakes, rivers, water_buffer_m)

    edges = edges_after_road.copy()
    if "added_by_road" not in edges.columns:
        edges["added_by_road"] = False
    if "removed_by_water" not in edges.columns:
        edges["removed_by_water"] = False

    # Compute ratio only for edges kept after road pruning
    kept_mask = edges.get("removed_by_road", False) == False
    cand = edges[kept_mask & edges.geometry.notnull()].copy()
    for idx, r in cand.iterrows():
        ratio = _boundary_water_ratio(r.geometry, unioned)
        edges.at[idx, "water_ratio"] = ratio
        if ratio >= float(water_threshold):
            # road exception: use same bridging test
            edges_sub = edges.loc[[idx]].copy()
            check = add_back_edges_via_roads(edges_sub, precincts, roads, road_boundary_buffer_m)
            if bool(check.iloc[0].get("added_by_road", False)):
                edges.at[idx, "added_by_road"] = True
                edges.at[idx, "removed_by_water"] = False
            else:
                edges.at[idx, "removed_by_water"] = True

    # For edges not evaluated set default water_ratio to 0
    edges["water_ratio"] = edges.get("water_ratio", pd.Series(0.0, index=edges.index)).fillna(0.0)
    return edges


# --- New: Step 3 manual add-back list ---

def apply_manual_addbacks(edges: gpd.GeoDataFrame, csv_path: Optional[str]) -> gpd.GeoDataFrame:
    if not csv_path or not Path(csv_path).exists():
        return edges
    df = pd.read_csv(csv_path)
    # normalize tuple order (min, max)
    add_pairs: set[Tuple[int, int]] = set()
    for _, r in df.iterrows():
        try:
            u = int(r["u"]) if "u" in r else int(r[0])
            v = int(r["v"]) if "v" in r else int(r[1])
        except Exception:
            continue
        a, b = (u, v) if u <= v else (v, u)
        add_pairs.add((a, b))

    edges = edges.copy()
    # Flip removal flags for matching edges; if edge not present, append it
    existing_pairs = set((min(int(r.u), int(r.v)), max(int(r.u), int(r.v))) for _, r in edges.iterrows())
    to_add_records: List[Dict] = []
    for a, b in add_pairs:
        if (a, b) in existing_pairs:
            mask = ((edges.u == a) & (edges.v == b)) | ((edges.u == b) & (edges.v == a))
            edges.loc[mask, "removed_by_water"] = False
            edges.loc[mask, "removed_by_road"] = False
            edges.loc[mask, "added_by_road"] = True
        else:
            to_add_records.append({"u": a, "v": b, "boundary_len": 0.0, "geometry": None, "removed_by_water": False, "removed_by_road": False, "added_by_road": True})

    if to_add_records:
        extra = gpd.GeoDataFrame(to_add_records, geometry="geometry", crs=edges.crs)
        edges = pd.concat([edges, extra], ignore_index=True)
    return edges


