from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, cast

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmium
import osmnx as ox
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point

LatLon = Tuple[float, float]
Coord = Tuple[float, float]

# =====================
# CONSTANTS
# =====================
INPUT_FILE = Path("data/stops.kmz")

PBF_FILES = [
    Path("data/nord-ovest-260331.osm.pbf"),
    Path("data/switzerland-260331.osm.pbf"),
]

OUTPUT_ROUTE = Path("data/journey_route.geojson")
OUTPUT_MAP = Path("data/journey_map.png")

EXPLODE_MULTIPOINTS = True

# Local railway extract extent
RAIL_BUFFER_DEG = 0.25

# Road fallback extent
ROAD_BUFFER_DEG = 0.15

# Snap tolerances
RAIL_SNAP_TOLERANCE_M = 15000
ROAD_SNAP_TOLERANCE_M = 5000

# Plot settings
PAD_METERS = 150000
FIGSIZE = (12, 12)
ROUTE_LINEWIDTH = 3
STOP_SIZE = 40
RAILWAY_BG_WIDTH = 1

COLOR = "red"

# Optional cache for OSMnx road queries
ox.settings.use_cache = True
ox.settings.log_console = False


def read_kml_or_kmz(input_file: Path, explode_multipoints: bool) -> gpd.GeoDataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    suffix = input_file.suffix.lower()

    if suffix == ".kml":
        gdf = gpd.read_file(input_file)
        return normalize_points_gdf(gdf, explode_multipoints)

    if suffix == ".kmz":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with zipfile.ZipFile(input_file, "r") as zf:
                zf.extractall(tmpdir_path)

            kml_files = list(tmpdir_path.rglob("*.kml"))
            if not kml_files:
                raise FileNotFoundError("No .kml file found inside the KMZ archive.")

            gdf = gpd.read_file(kml_files[0])
            return normalize_points_gdf(gdf, explode_multipoints)

    raise ValueError("Input file must be .kml or .kmz")


def normalize_points_gdf(
    gdf: gpd.GeoDataFrame,
    explode_multipoints: bool,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        raise ValueError("Input layer is empty.")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    rows = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        base = row.drop(labels=["geometry"]).to_dict()

        if isinstance(geom, Point):
            rows.append({**base, "source_index": idx, "geometry": geom})

        elif isinstance(geom, MultiPoint):
            for part_idx, pt in enumerate(geom.geoms):
                rows.append(
                    {
                        **base,
                        "source_index": idx,
                        "multipart_index": part_idx,
                        "geometry": pt,
                    }
                )

    if not rows:
        raise ValueError("No point geometries found in the input file.")

    out = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    out["journey_order"] = range(1, len(out) + 1)
    return out


def extract_ordered_points(points_gdf: gpd.GeoDataFrame) -> List[LatLon]:
    ordered: List[LatLon] = []

    for geom in points_gdf.geometry:
        if isinstance(geom, Point):
            ordered.append((geom.y, geom.x))

    if len(ordered) < 2:
        raise ValueError("At least two points are required.")

    return ordered


def get_bounds(
    points: List[LatLon], buffer_deg: float
) -> tuple[float, float, float, float]:
    lats = [lat for lat, _ in points]
    lons = [lon for _, lon in points]

    north = max(lats) + buffer_deg
    south = min(lats) - buffer_deg
    east = max(lons) + buffer_deg
    west = min(lons) - buffer_deg

    return north, south, east, west


def fetch_railways(points: List[LatLon], buffer_deg: float) -> gpd.GeoDataFrame:
    north, south, east, west = get_bounds(points, buffer_deg)

    allowed_railways = {"rail", "light_rail", "subway", "tram", "narrow_gauge"}
    rows: List[dict] = []

    for pbf_file in PBF_FILES:
        if not pbf_file.exists():
            raise FileNotFoundError(f"PBF file not found: {pbf_file}")

        class RailwayHandler(osmium.SimpleHandler):
            def way(self, w: osmium.osm.Way) -> None:
                railway = w.tags.get("railway")
                if railway not in allowed_railways:
                    return

                try:
                    coords = [(node.lon, node.lat) for node in w.nodes]
                except Exception:
                    return

                if len(coords) < 2:
                    return

                xs = [x for x, _ in coords]
                ys = [y for _, y in coords]

                if (
                    max(xs) < west
                    or min(xs) > east
                    or max(ys) < south
                    or min(ys) > north
                ):
                    return

                rows.append(
                    {
                        "source_pbf": str(pbf_file),
                        "osm_id": int(w.id),
                        "railway": railway,
                        "geometry": LineString(coords),
                    }
                )

        handler = RailwayHandler()
        handler.apply_file(str(pbf_file), locations=True)

    if not rows:
        raise ValueError("No railway features found in the requested area.")

    rail = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    rail = rail[rail.geometry.notna()].copy()
    rail = rail[rail.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    assert type(rail) is gpd.GeoDataFrame

    if rail.empty:
        raise ValueError("No linear railway geometries found in the requested area.")

    return rail


def build_road_graph(points: List[LatLon], buffer_deg: float) -> nx.Graph:
    north, south, east, west = get_bounds(points, buffer_deg)

    allowed_highways = {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
    }

    G = nx.Graph()

    for pbf_file in PBF_FILES:
        if not pbf_file.exists():
            raise FileNotFoundError(f"PBF file not found: {pbf_file}")

        class RoadHandler(osmium.SimpleHandler):
            def way(self, w: osmium.osm.Way) -> None:
                highway = w.tags.get("highway")
                if highway not in allowed_highways:
                    return

                try:
                    coords = [(node.lon, node.lat) for node in w.nodes]
                except Exception:
                    return

                if len(coords) < 2:
                    return

                xs = [x for x, _ in coords]
                ys = [y for _, y in coords]

                if (
                    max(xs) < west
                    or min(xs) > east
                    or max(ys) < south
                    or min(ys) > north
                ):
                    return

                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]

                    n1 = (round(x1, 7), round(y1, 7))
                    n2 = (round(x2, 7), round(y2, 7))

                    segment = LineString([(x1, y1), (x2, y2)])
                    length = segment.length

                    if n1 == n2 or length == 0:
                        continue

                    G.add_edge(n1, n2, weight=length)

        handler = RoadHandler()
        handler.apply_file(str(pbf_file), locations=True)

    if G.number_of_edges() == 0:
        raise ValueError("No road graph built.")

    return G


def explode_lines(rail_gdf: gpd.GeoDataFrame) -> List[LineString]:
    lines: List[LineString] = []

    for geom in rail_gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        if isinstance(geom, LineString):
            if len(geom.coords) >= 2:
                lines.append(geom)

        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                if len(part.coords) >= 2:
                    lines.append(part)

    if not lines:
        raise ValueError("No usable railway line segments found.")

    return lines


def rounded_coord(x: float, y: float, ndigits: int = 7) -> Coord:
    return (round(x, ndigits), round(y, ndigits))


def build_rail_graph(lines: List[LineString]) -> nx.Graph:
    G = nx.Graph()

    for line in lines:
        coords = list(line.coords)

        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]

            n1 = rounded_coord(x1, y1)
            n2 = rounded_coord(x2, y2)

            segment = LineString([(x1, y1), (x2, y2)])
            length = segment.length

            if n1 == n2 or length == 0:
                continue

            G.add_node(n1, x=n1[0], y=n1[1])
            G.add_node(n2, x=n2[0], y=n2[1])

            if G.has_edge(n1, n2):
                if length < G[n1][n2]["weight"]:
                    G[n1][n2]["weight"] = length
                    G[n1][n2]["geometry"] = segment
            else:
                G.add_edge(n1, n2, weight=length, geometry=segment)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise ValueError("Railway graph could not be built.")

    return G


def nearest_graph_node(
    point_xy: Coord,
    node_coords_3857: Dict[Coord, Coord],
) -> Tuple[Coord, float]:
    px, py = point_xy

    best_node = None
    best_dist_sq = None

    for node, (nx_, ny_) in node_coords_3857.items():
        dx = nx_ - px
        dy = ny_ - py
        d2 = dx * dx + dy * dy

        if best_dist_sq is None or d2 < best_dist_sq:
            best_dist_sq = d2
            best_node = node

    if best_node is None or best_dist_sq is None:
        raise ValueError("Could not find nearest graph node.")

    return best_node, best_dist_sq**0.5


def concatenate_path_geometries(path_nodes: List[Coord]) -> LineString:
    coords: List[Coord] = []

    for i in range(len(path_nodes) - 1):
        a = path_nodes[i]
        b = path_nodes[i + 1]

        if i == 0:
            coords.append(a)
        coords.append(b)

    return LineString(coords)


def nearest_node_bruteforce(point: Point, nodes: List[Coord]) -> Coord:
    px, py = point.x, point.y

    best = None
    best_d = None

    for nx_, ny_ in nodes:
        dx = nx_ - px
        dy = ny_ - py
        d = dx * dx + dy * dy

        if best_d is None or d < best_d:
            best_d = d
            best = (nx_, ny_)

    if best is None:
        raise ValueError("Could not find nearest road node.")

    return best


def build_bus_route_segment(
    G_drive: nx.Graph,
    start_geom: Point,
    end_geom: Point,
    snap_tolerance_m: float,
) -> tuple[LineString, float, float]:
    nodes = list(G_drive.nodes)

    start_node = nearest_node_bruteforce(start_geom, nodes)
    end_node = nearest_node_bruteforce(end_geom, nodes)

    path = nx.shortest_path(G_drive, start_node, end_node, weight="weight")
    coords = [(x, y) for (x, y) in path]

    return LineString(coords), 0.0, 0.0


def build_mixed_journey_route(
    points_gdf: gpd.GeoDataFrame,
    rail_gdf: gpd.GeoDataFrame,
    ordered_points: List[LatLon],
    rail_snap_tolerance_m: float,
    road_snap_tolerance_m: float,
    road_buffer_deg: float,
) -> gpd.GeoDataFrame:
    rail_lines = explode_lines(rail_gdf)
    G_rail = build_rail_graph(rail_lines)
    G_drive = build_road_graph(ordered_points, road_buffer_deg)

    points_3857 = points_gdf.to_crs(epsg=3857)

    rail_nodes_gdf = gpd.GeoDataFrame(
        [{"node": n, "geometry": Point(n[0], n[1])} for n in G_rail.nodes],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    node_coords_3857: Dict[Coord, Coord] = {}

    for _, row in rail_nodes_gdf.iterrows():
        node = cast(Coord, row["node"])
        coord_3857 = (float(row.geometry.x), float(row.geometry.y))
        node_coords_3857[node] = coord_3857

    snapped_rail_nodes: List[Coord | None] = []
    snapped_rail_dists: List[float | None] = []

    for _, row in points_3857.iterrows():
        pxy = (row.geometry.x, row.geometry.y)
        node, dist = nearest_graph_node(pxy, node_coords_3857)

        if dist <= rail_snap_tolerance_m:
            snapped_rail_nodes.append(node)
            snapped_rail_dists.append(float(dist))
        else:
            snapped_rail_nodes.append(None)
            snapped_rail_dists.append(float(dist))

    original_points = list(points_gdf.geometry)
    segments = []

    for i in range(len(original_points) - 1):
        start_geom = original_points[i]
        end_geom = original_points[i + 1]

        if not isinstance(start_geom, Point) or not isinstance(end_geom, Point):
            continue

        start_rail_node = snapped_rail_nodes[i]
        end_rail_node = snapped_rail_nodes[i + 1]

        start_rail_dist = snapped_rail_dists[i]
        end_rail_dist = snapped_rail_dists[i + 1]

        segment_mode = None
        line = None
        start_snap_dist_m = None
        end_snap_dist_m = None

        if start_rail_node is not None and end_rail_node is not None:
            try:
                path_nodes = nx.shortest_path(
                    G_rail,
                    start_rail_node,
                    end_rail_node,
                    weight="weight",
                )
                line = concatenate_path_geometries(path_nodes)
                segment_mode = "train"
                start_snap_dist_m = start_rail_dist
                end_snap_dist_m = end_rail_dist
            except nx.NetworkXNoPath:
                pass

        if line is None:
            try:
                line, road_start_dist, road_end_dist = build_bus_route_segment(
                    G_drive=G_drive,
                    start_geom=start_geom,
                    end_geom=end_geom,
                    snap_tolerance_m=road_snap_tolerance_m,
                )
                segment_mode = "bus"
                start_snap_dist_m = road_start_dist
                end_snap_dist_m = road_end_dist
            except Exception:
                pass

        if line is None:
            line = LineString(
                [
                    (start_geom.x, start_geom.y),
                    (end_geom.x, end_geom.y),
                ]
            )
            segment_mode = "straight"
            start_snap_dist_m = None
            end_snap_dist_m = None

        segments.append(
            {
                "segment_id": i + 1,
                "from_point": i + 1,
                "to_point": i + 2,
                "mode": segment_mode,
                "start_snap_dist_m": start_snap_dist_m,
                "end_snap_dist_m": end_snap_dist_m,
                "geometry": line,
            }
        )

    if not segments:
        raise ValueError("No journey segments could be built.")

    return gpd.GeoDataFrame(segments, crs="EPSG:4326")


def write_output(route_gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    route_gdf.to_file(output_path, driver="GeoJSON")


def plot_map(
    route_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    rail_gdf: gpd.GeoDataFrame,
) -> None:
    route_3857 = route_gdf.to_crs(epsg=3857)
    points_3857 = points_gdf.to_crs(epsg=3857)
    rail_3857 = rail_gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    rail_3857.plot(ax=ax, linewidth=RAILWAY_BG_WIDTH, alpha=0.3, color=COLOR)

    train = route_3857[route_3857["mode"] == "train"]
    bus = route_3857[route_3857["mode"] == "bus"]
    straight = route_3857[route_3857["mode"] == "straight"]

    if not train.empty:
        train.plot(ax=ax, linewidth=ROUTE_LINEWIDTH, color=COLOR)

    if not bus.empty:
        bus.plot(ax=ax, linewidth=ROUTE_LINEWIDTH, color=COLOR)

    if not straight.empty:
        straight.plot(ax=ax, linewidth=ROUTE_LINEWIDTH, alpha=0.7, color=COLOR)

    points_3857.plot(ax=ax, markersize=STOP_SIZE, color=COLOR)

    for _, row in points_3857.iterrows():
        ax.text(row.geometry.x, row.geometry.y, str(row["journey_order"]), fontsize=9)

    minx, miny, maxx, maxy = route_3857.total_bounds
    ax.set_xlim(minx - PAD_METERS, maxx + PAD_METERS)
    ax.set_ylim(miny - PAD_METERS, maxy + PAD_METERS)

    ctx.add_basemap(ax)
    ax.set_axis_off()

    plt.savefig(OUTPUT_MAP, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        points_gdf = read_kml_or_kmz(INPUT_FILE, EXPLODE_MULTIPOINTS)
        ordered_points = extract_ordered_points(points_gdf)

        rail_gdf = fetch_railways(ordered_points, RAIL_BUFFER_DEG)

        route_gdf = build_mixed_journey_route(
            points_gdf=points_gdf,
            rail_gdf=rail_gdf,
            ordered_points=ordered_points,
            rail_snap_tolerance_m=RAIL_SNAP_TOLERANCE_M,
            road_snap_tolerance_m=ROAD_SNAP_TOLERANCE_M,
            road_buffer_deg=ROAD_BUFFER_DEG,
        )

        write_output(route_gdf, OUTPUT_ROUTE)
        plot_map(route_gdf, points_gdf, rail_gdf)

        print(f"Loaded {len(points_gdf)} ordered points.")
        print(f"Fetched {len(rail_gdf)} railway features.")
        print(f"Created {len(route_gdf)} journey segment(s).")
        print(f"Modes used: {sorted(route_gdf['mode'].unique().tolist())}")
        print(f"PBF sources: {[str(p) for p in PBF_FILES]}")
        print(f"Saved route to: {OUTPUT_ROUTE.resolve()}")
        print(f"Saved map to: {OUTPUT_MAP.resolve()}")
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
