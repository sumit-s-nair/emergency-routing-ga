"""
Dataset generation / OSM fetch placeholder.

This module will later fetch map-based datasets using the OpenStreetMap API or
`osmnx` and store them under `datasets/`. For now it provides a stub.
"""


def fetch_osm_bbox(north, south, east, west, out_path=None):
    """Fetch a bounding-box extract from OSM and write to out_path (placeholder).

    Args:
        north, south, east, west: float bounding box coordinates
        out_path: path to save dataset
    Returns:
        path to saved dataset or None
    """
    # TODO: implement using `osmnx.graph_from_bbox` or Overpass API
    print("fetch_osm_bbox called with:", north, south, east, west)
    if out_path:
        print(f"(placeholder) would save dataset to {out_path}")
    return out_path
