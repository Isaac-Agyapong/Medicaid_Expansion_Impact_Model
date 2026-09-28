"""Generate the app's hero artwork: a blue gradient with a glowing dot map of every US county
(orange dots = counties in the 10 states that have not expanded Medicaid).

    python app/make_assets.py      (writes app/assets/hero.png, committed)
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent / "assets" / "hero.png"
GEOJSON = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
W, H = 2240, 1100            # about the banner's shape on a laptop screen
TOP, BOTTOM = (15, 30, 82), (37, 78, 214)          # deep navy to royal blue


def albers(lon, lat, lon0=-96, lat0=37.5, p1=29.5, p2=45.5):
    lon, lat, lon0, lat0, p1, p2 = map(np.radians, (lon, lat, lon0, lat0, p1, p2))
    n = (np.sin(p1) + np.sin(p2)) / 2
    c = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho = np.sqrt(c - 2 * n * np.sin(lat)) / n
    rho0 = np.sqrt(c - 2 * n * np.sin(lat0)) / n
    return rho * np.sin(n * (lon - lon0)), rho0 - rho * np.cos(n * (lon - lon0))


def main():
    with urllib.request.urlopen(GEOJSON, timeout=60) as r:
        geo = json.loads(r.read())
    c = pd.read_csv(Path(__file__).resolve().parents[1] / "models" / "app_counties.csv", dtype={"county_fips": str})
    # the 10 states that still have not expanded (NC and SD expanded in late 2023)
    remaining = set(c.loc[c.expansion_year.isna() & ~c.state.isin(["NC", "SD"]), "county_fips"])
    pts, warm_flags = [], []
    for f in geo["features"]:
        if f["id"][:2] in ("02", "15", "72"):
            continue
        g = f["geometry"]
        ring = g["coordinates"][0] if g["type"] == "Polygon" else max(g["coordinates"], key=lambda p: len(p[0]))[0]
        xy = np.array(ring)
        pts.append(albers(xy[:, 0].mean(), xy[:, 1].mean()))
        warm_flags.append(f["id"] in remaining)
    pts = np.array(pts)

    # diagonal gradient
    yy, xx = np.mgrid[0:H, 0:W]
    t = (0.65 * xx / W + 0.35 * yy / H)[..., None]
    img = Image.fromarray((np.array(TOP) * (1 - t) + np.array(BOTTOM) * t).astype(np.uint8))

    # county dots on the right half, brighter towards the south-east (a hint of the coverage gap)
    x, y = pts[:, 0], pts[:, 1]
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    left, right, top, bottom = 1260, 2180, 150, 740
    px, py = left + x * (right - left), bottom - y * (bottom - top)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dots = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dg, dd = ImageDraw.Draw(glow), ImageDraw.Draw(dots)
    for a, b, warm in zip(px, py, warm_flags):                     # orange = states that have not expanded
        colour = (255, 170, 90) if warm else (190, 210, 255)
        alpha = 230 if warm else 150
        dd.ellipse([a - 3.2, b - 3.2, a + 3.2, b + 3.2], fill=(*colour, alpha))
        dg.ellipse([a - 7, b - 7, a + 7, b + 7], fill=(*colour, 60 if warm else 30))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(6)))
    img = Image.alpha_composite(img, dots)

    # soft fade on the left so the headline stays readable
    fade = np.zeros((H, W), dtype=np.uint8)
    fade[:, :1300] = np.clip((1300 - np.arange(1300)) / 1300 * 120, 0, 120).astype(np.uint8)
    shade = Image.new("RGBA", (W, H), (*TOP, 0))
    shade.putalpha(Image.fromarray(fade))
    img = Image.alpha_composite(img, shade)

    OUT.parent.mkdir(exist_ok=True)
    img.convert("RGB").save(OUT, optimize=True, quality=90)
    print(f"wrote {OUT} ({len(pts):,} counties)")


if __name__ == "__main__":
    main()
