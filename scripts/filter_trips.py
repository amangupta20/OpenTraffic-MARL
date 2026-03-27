#!/usr/bin/env python3
"""
Filter Bangalore MG Road trip files to:
  1. Remove trips that spawn/terminate on edges deep inside internal compounds
     (areas with no traffic lights — they cause uncontrollable gridlock).
  2. Apply an overall traffic density reduction factor.

Usage (run inside the container):
  python3 scripts/filter_trips.py [--density-factor 0.6] [--dry-run]

Output: writes cleaned trip files alongside originals as *_cleaned.xml
The bangalore_corridor env is then pointed at the cleaned files via sumocfg.
"""

import argparse
import gzip
import pathlib
import shutil
import xml.etree.ElementTree as ET

NET_DIR = pathlib.Path("/app/sumo_net/bangalore_mg_road")
NET_FILE = NET_DIR / "osm.net.xml.gz"

# Trip files to process (pedestrians have their own rou.xml handled separately)
TRIP_FILES = [
    "osm.passenger.trips.xml",
    "osm.motorcycle.trips.xml",
    "osm.truck.trips.xml",
    "osm.bicycle.trips.xml",
    "osm.bus.trips.xml",
]


def load_tls_controlled_edges(net_path: pathlib.Path) -> set[str]:
    """
    Parse the net XML to find all edge IDs that are incoming/outgoing at a
    junction which has a traffic light (type='traffic_light' or 'traffic_light_unregulated').
    """
    print("[filter] Parsing network for TLS-controlled junctions...")
    tls_node_ids: set[str] = set()
    tls_adjacent_edges: set[str] = set()

    with gzip.open(net_path, "rb") as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # Collect junction IDs that have TLS
    for junction in root.iter("junction"):
        j_type = junction.get("type", "")
        if "traffic_light" in j_type:
            tls_node_ids.add(junction.get("id"))

    print(f"[filter] Found {len(tls_node_ids)} TLS junctions.")

    # Collect edges connected to those junctions
    for edge in root.iter("edge"):
        edge_id = edge.get("id", "")
        if edge_id.startswith(":"):  # internal edge — skip
            continue
        from_node = edge.get("from", "")
        to_node   = edge.get("to", "")
        if from_node in tls_node_ids or to_node in tls_node_ids:
            tls_adjacent_edges.add(edge_id)
            # Also include the reverse edge (negative prefix convention)
            tls_adjacent_edges.add(f"-{edge_id}")

    print(f"[filter] {len(tls_adjacent_edges)} edges are adjacent to TLS junctions.")
    return tls_adjacent_edges


def find_reachable_edges(net_path: pathlib.Path, tls_edges: set[str]) -> set[str]:
    """
    BFS from TLS-adjacent edges through the edge adjacency graph to find all
    edges reachable within N hops. This captures the full main corridor.
    Edges NOT reachable are the internal dead-end compounds.
    """
    MAX_HOPS = 4  # Allow up to 4 edges away from any TLS junction

    print(f"[filter] Building adjacency graph (max {MAX_HOPS} hops from TLS)...")

    with gzip.open(net_path, "rb") as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # Map: to_node -> list of edge_ids that arrive there
    # Map: from_node -> list of edge_ids that depart from there
    node_outgoing: dict[str, list[str]] = {}
    node_incoming: dict[str, list[str]] = {}
    edge_from: dict[str, str] = {}
    edge_to:   dict[str, str] = {}

    # Identify parking lanes and small service roads to explicitly exclude
    restricted_types = {"highway.service", "highway.residential", "highway.unclassified", "highway.cycleway", "highway.footway"}
    restricted_edges: set[str] = set()

    for edge in root.iter("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        
        edge_type = edge.get("type", "")
        if edge_type in restricted_types:
            restricted_edges.add(eid)
            
        fn = edge.get("from", "")
        tn = edge.get("to", "")
        edge_from[eid] = fn
        edge_to[eid]   = tn
        node_outgoing.setdefault(fn, []).append(eid)
        node_incoming.setdefault(tn, []).append(eid)

    # BFS
    visited: set[str] = set(tls_edges)
    frontier = set(tls_edges)
    for hop in range(MAX_HOPS):
        next_frontier: set[str] = set()
        for eid in frontier:
            # Walk forward (through edge's to-node)
            to_node = edge_to.get(eid) or edge_to.get(eid.lstrip("-"), "")
            for neighbour in node_outgoing.get(to_node, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
            # Walk backward (through edge's from-node)
            from_node = edge_from.get(eid) or edge_from.get(eid.lstrip("-"), "")
            for neighbour in node_incoming.get(from_node, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.add(neighbour)
        frontier = next_frontier
        print(f"[filter]   hop {hop+1}: {len(visited)} reachable edges")

    # Remove the restricted edges completely from the allowed list
    filtered_visited = visited - restricted_edges
    print(f"[filter] Excluded {len(restricted_edges)} parking/service/residential edges.")
    return filtered_visited


def filter_trip_file(
    src: pathlib.Path,
    allowed_edges: set[str],
    density_factor: float,
    dry_run: bool,
) -> tuple[int, int]:
    """Filter a single trip file. Returns (original_count, kept_count)."""
    tree = ET.parse(src)
    root = tree.getroot()

    trips = root.findall("trip")
    flows = root.findall("flow")
    elements = trips + flows

    original = len(elements)
    kept = 0
    removed_edge = 0

    # Apply density factor by striding (keep every 1/density_factor-th trip)
    stride = max(1, round(1.0 / density_factor))

    for i, el in enumerate(elements):
        from_edge = el.get("from", "")
        to_edge   = el.get("to", "")

        # Remove if either endpoint is not reachable from TLS network
        if from_edge not in allowed_edges or to_edge not in allowed_edges:
            root.remove(el)
            removed_edge += 1
            continue

        # Density thinning: remove every (stride-th+1) trip
        if stride > 1 and (i % stride) != 0:
            root.remove(el)
            continue

        kept += 1

    dst = src.parent / src.name.replace(".xml", "_cleaned.xml")
    if not dry_run:
        ET.indent(root, space="    ")
        tree.write(str(dst), encoding="unicode", xml_declaration=True)

    print(f"[filter] {src.name}: {original} → {kept} trips kept "
          f"({removed_edge} removed for internal compound, "
          f"{original - removed_edge - kept} thinned for density)")
    return original, kept


def patch_sumocfg(net_dir: pathlib.Path, dry_run: bool) -> None:
    """Rewrite osm.sumocfg to point at _cleaned trip files and add flow controls."""
    cfg_src = net_dir / "osm.sumocfg"
    cfg_dst = net_dir / "osm_cleaned.sumocfg"

    tree = ET.parse(cfg_src)
    root = tree.getroot()

    # Update route-files to use cleaned versions
    inp = root.find("input")
    if inp is not None:
        rf = inp.find("route-files")
        if rf is not None:
            orig = rf.get("value", "")
            cleaned = orig.replace(".trips.xml", ".trips_cleaned.xml") \
                          .replace(".rou.xml", ".rou.xml")  # keep pedestrian rou.xml
            rf.set("value", cleaned)

    # Add processing options for smoother flow
    proc = root.find("processing")
    if proc is None:
        proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "max-depart-delay").set("value", "300")     # kill after 5min wait
    ET.SubElement(proc, "time-to-teleport").set("value", "200")     # teleport after ~3min stuck
    ET.SubElement(proc, "random-depart-offset").set("value", "60")  # spread arrivals ±60s

    if not dry_run:
        ET.indent(root, space="    ")
        tree.write(str(cfg_dst), encoding="unicode", xml_declaration=True)
        print(f"[filter] Wrote {cfg_dst.name}")
    else:
        print(f"[filter] (dry-run) Would write {cfg_dst.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-factor", type=float, default=0.65,
                        help="Fraction of trips to keep (default 0.65 = 35% reduction)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    print(f"[filter] density_factor={args.density_factor}  dry_run={args.dry_run}")

    # Step 1: find all edges adjacent to TLS junctions
    tls_edges = load_tls_controlled_edges(NET_FILE)

    # Step 2: BFS to find all reachable main-corridor edges
    allowed_edges = find_reachable_edges(NET_FILE, tls_edges)
    print(f"[filter] Total allowed edges (main corridor): {len(allowed_edges)}")

    # Step 3: filter each trip file
    total_orig, total_kept = 0, 0
    for fname in TRIP_FILES:
        src = NET_DIR / fname
        if not src.exists():
            print(f"[filter] Skipping {fname} (not found)")
            continue
        o, k = filter_trip_file(src, allowed_edges, args.density_factor, args.dry_run)
        total_orig += o
        total_kept += k

    # Step 4: patch sumocfg
    patch_sumocfg(NET_DIR, args.dry_run)

    print(f"\n[filter] Done. Total trips: {total_orig} → {total_kept} "
          f"({100*(1-total_kept/total_orig):.1f}% reduction)")
    if not args.dry_run:
        print("[filter] Cleaned files written. Update bangalore_corridor.py to use osm_cleaned.sumocfg")


if __name__ == "__main__":
    main()
