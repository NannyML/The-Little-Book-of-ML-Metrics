"""Preview annotated formulas exactly as the book sets them, and measure where their symbols are.

    uv run python tools/formula_preview.py OUT.png "5-ranking.tex:@K Metrics" "8-genai.tex:FID" ...
    uv run python tools/formula_preview.py --positions "8-genai.tex:FID"
    uv run python tools/formula_preview.py --dpi 300 OUT.png "8-genai.tex:FID"     # sharper sheet to check clearances

The sheet shows each formula at book scale inside the text block (gray box) with the text block's
center line (red), so you can judge arrow placement, label collisions and whether the math itself sits
in the middle of the page. The environment centers the whole drawing, labels included, so a long label on
one side pushes the formula to the other.

--positions renders the formula without its arrows and prints every colored symbol's x position and how far
its top lies below a.north and its bottom above a.south, in cm: the numbers to use in
\\draw (\\$(a.north)+(x,-dy)\\$) ... Symbols of the second line in a two-node block are measured in the same frame.
"""
import io
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formula_tuner as ft  # noqa: E402

BOOK = ft.BOOK
COLORS = {"red": (221, 64, 64), "cyan": (0, 200, 229), "purple": (59, 2, 128), "green": (78, 176, 70), "yellow": (225, 188, 41)}


def block(spec):
    chapter, section = spec.split(":", 1)
    s = (BOOK / chapter).read_text()
    m = re.search(r"\\(?:metricentry|monitorentry)\{" + re.escape(section) + r"\}", s)
    if not m:
        sys.exit(f"no section {section!r} in {chapter}")
    a = s.index("\\begin{metricformula}", m.start())
    b = s.index("\\end{metricformula}", a) + len("\\end{metricformula}")
    return section, s[a:b]


def text_block():
    png, _ = ft.render("\\noindent\\rule{\\linewidth}{4pt}")
    a = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    xs = np.nonzero(a < 100)[1]
    return xs.min(), xs.max()


def sheet(out, specs):
    font = ImageFont.truetype(str(BOOK / "fonts/Inter/Inter-SemiBold.otf"), 20)
    x0, x1 = text_block()
    rows = []
    for spec in specs:
        title, tex = block(spec)
        png, err = ft.render(tex)
        if err:
            print("ERROR", spec, err[-400:])
            continue
        im = Image.open(io.BytesIO(png)).convert("RGB")
        a = np.asarray(im)
        ys, xs = np.nonzero((a < 245).any(axis=2))
        im = im.crop((x0 - 60, ys.min() - 12, x1 + 60, ys.max() + 12))
        d = ImageDraw.Draw(im)
        d.rectangle([60, 0, 60 + x1 - x0, im.height - 1], outline=(200, 200, 200))
        d.line([(60 + (x1 - x0) // 2, 0), (60 + (x1 - x0) // 2, im.height)], fill=(255, 120, 120))
        if xs.max() > x1 + 2 or xs.min() < x0 - 2:
            title += "  (wider than the text block: scaled down)"
        rows.append((title, im))
    W = max(im.width for _, im in rows) + 20
    out_im = Image.new("RGB", (W, sum(im.height + 40 for _, im in rows) + 10), "white")
    d = ImageDraw.Draw(out_im)
    y = 6
    for title, im in rows:
        d.text((10, y), title, fill=(200, 80, 0), font=font)
        out_im.paste(im, (10, y + 30))
        y += im.height + 40
    out_im.save(out)
    print("wrote", out)


def positions(spec):
    title, tex = block(spec)
    bare = "\n".join(l for l in tex.splitlines() if not re.match(r"\s*\\(draw|path)\b", l))
    marks = "".join(f"\\fill[overlay, color={{rgb,255:red,{r};green,{g};blue,{b}}}] (a.{anc}) ++(-0.8pt,-0.8pt) rectangle ++(1.6pt,1.6pt);\n"
                    for anc, (r, g, b) in [("north", (255, 0, 255)), ("south", (0, 255, 0)), ("west", (255, 128, 0)), ("east", (0, 128, 255))])
    k = bare.rindex("}", 0, bare.index("\\end{metricformula}"))
    png, err = ft.render(bare[:k] + marks + bare[k:])
    if err:
        sys.exit(err[-400:])
    im = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)

    def mark(rgb):
        ys, xs = np.nonzero((abs(im - np.array(rgb)) < 8).all(axis=2))
        return xs.mean(), ys.mean()

    (nx, ny), (sx, sy), (wx, _), (ex, _) = (mark(c) for c in [(255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255)])
    cm = ft.DPI / 2.54
    print(f"{title}: node a {(ex - wx) / cm:.2f} x {(sy - ny) / cm:.2f} cm, west {(wx - nx) / cm:+.2f}, east {(ex - nx) / cm:+.2f}")
    for name, rgb in COLORS.items():
        hit = (abs(im - np.array(rgb)) < 40).all(axis=2)
        lab, _ = ndimage.label(ndimage.binary_dilation(hit, iterations=3))
        for sl in ndimage.find_objects(lab):
            ya, yb, xa, xb = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            if (xb - xa) * (yb - ya) < 30:
                continue
            print(f"  {name:7s} x {((xa + xb) / 2 - nx) / cm:+.2f}  ({(xa - nx) / cm:+.2f} to {(xb - nx) / cm:+.2f})"
                  f"   top {(ya - ny) / cm:.2f} below a.north   bottom {(sy - yb) / cm:+.2f} above a.south")


if __name__ == "__main__":
    ft.ensure_format()
    if sys.argv[1] == "--dpi":
        ft.DPI = int(sys.argv[2])
        sys.argv[1:3] = []
    if sys.argv[1] == "--positions":
        for spec in sys.argv[2:]:
            positions(spec)
    else:
        sheet(sys.argv[1], sys.argv[2:])
