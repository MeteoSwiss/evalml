"""Extract a single Swiss canton boundary into a shapefile.

Downloads GADM level-1 boundaries for Switzerland, keeps one canton and writes it
as a shapefile in EPSG:2056, matching the default src_crs of
verification.ShapefileSpatialAggregationMasks.
"""

import logging
from argparse import ArgumentParser

import geopandas as gpd

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# GADM 4.1 level-1 = Swiss cantons (field NAME_1, e.g. "Valais"). WGS84 source.
GADM_CHE_LEVEL1_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_CHE_1.json"
LV95 = "EPSG:2056"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canton",
        default="Valais",
        help="Canton name as in GADM NAME_1 (default: Valais).",
    )
    parser.add_argument(
        "--url",
        default=GADM_CHE_LEVEL1_URL,
        help="Source of Swiss canton boundaries (GADM level-1).",
    )
    parser.add_argument("--outfn", required=True, help="Output shapefile path (.shp).")
    args = parser.parse_args()

    LOG.info("Downloading cantons from %s", args.url)
    cantons = gpd.read_file(args.url)

    sel = cantons[cantons["NAME_1"] == args.canton]
    if sel.empty:
        available = sorted(cantons["NAME_1"].unique())
        raise ValueError(f"Canton {args.canton!r} not found. Available: {available}")

    sel = sel.to_crs(LV95)
    sel.to_file(args.outfn)
    LOG.info("Wrote %s (%d feature(s), CRS=%s)", args.outfn, len(sel), sel.crs)


if __name__ == "__main__":
    main()
