"""Automatic arrow layout for the book's annotated formulas, validated on the rendered output.

    uv run python tools/formula_layout.py 3-classification.tex [--skip "Confusion Matrix,FPR"] [--only "MCC"] [--write]

Without --write nothing is changed: a before/after sheet is written to --out (default: scratchpad)
so the result can be checked by eye. With --write the new arrows are saved into the chapter file
(a .tex.bak is kept), through the same code path as the tuner.

How it decides, per formula:
  1. render the formula without arrows and find every colored symbol by its color (the book's five
     annotation colors); each arrow's target is the symbol of its color closest to where the arrow
     started before;
  2. start the arrow 3 px outside the symbol, upward if the symbol sits in the top half of the
     formula and downward otherwise, unless that path runs through other ink, in which case the
     other direction is used;
  3. send the arrow outward (left if the symbol is left of center, right otherwise; a label goes
     straight up or down only when it would otherwise run off the page, since vertical labels cost
     page height) and bend it so it leaves the symbol close to vertically:
     down-left and up-right bend left, down-right and up-left bend right;
  4. put the label just outside the formula, then render the labels alone, measure them, and push
     apart any that overlap each other or the formula, or that run off the text area;
  5. keep everything else (label text, colors, line style) as it was.
"""
import argparse
import base64
import io
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formula_tuner as ft  # noqa: E402

COLORS = {"nmlred": (221, 64, 64), "nmlcyan": (0, 200, 229), "nmlpurple": (59, 2, 128), "nmlgreen": (78, 176, 70), "nmlyellow": (225, 188, 41)}
TOL = 60                 # per-channel tolerance when matching a symbol's color
GAP = 3                  # px between a symbol and the arrow's start
LABEL_CLEAR_CM = 0.30    # vertical room between the ink under/over the symbol and its label line
ROW_CM = 0.42            # extra room per stacked label row
NEAR_CM = 1.3            # symbols closer than this on the same side are treated as neighbors
OUT_CM = 1.0             # horizontal reach of an arrow, in cm
PAGE_W_PX = int(5.7 * ft.DPI)
MARGIN_PX = int(1.25 / 2.54 * ft.DPI) + 4       # text area, with a little slack


def b64img(png_b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")).astype(int)


def color_of(opts):
    for name in COLORS:
        if name in opts:
            return name
    return None


def components(im, rgb, merge=5, min_px=6):
    """Bounding boxes of connected blobs of a color (nearby blobs merged, so subscripts join their symbol)."""
    mask = np.all(np.abs(im - np.array(rgb)) < TOL, axis=2)
    if not mask.any():
        return []
    grown = ndimage.binary_dilation(mask, iterations=merge)
    lab, n = ndimage.label(grown)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(mask & (lab == i))
        if len(xs) >= min_px:
            out.append({"x0": int(xs.min()), "x1": int(xs.max()), "y0": int(ys.min()), "y1": int(ys.max()),
                        "cx": float(xs.mean()), "cy": float(ys.mean()), "n": int(len(xs))})
    return out


def node_boxes(anchors):
    boxes = {}
    for node in ("a", "b"):
        if f"{node}.west" in anchors:
            boxes[node] = {"x0": anchors[f"{node}.west"][0], "x1": anchors[f"{node}.east"][0],
                           "y0": anchors[f"{node}.north"][1], "y1": anchors[f"{node}.south"][1]}
    return boxes


def ink(im):
    return np.any(im < 200, axis=2)


def path_blocked(inkmask, sym, node, direction):
    """Is there other ink between the symbol and the formula's edge in that direction?"""
    x0, x1 = max(sym["x0"] - 1, 0), sym["x1"] + 2
    if direction == "up":
        band = inkmask[int(node["y0"]):max(int(node["y0"]), sym["y0"] - GAP), x0:x1]
    else:
        band = inkmask[sym["y1"] + GAP:int(node["y1"]) + 1, x0:x1]
    return band.size > 0 and band.sum() > 12


def plan(block, base, arrows):
    """New parameters for every arrow of a block. Returns (arrows, notes)."""
    im = b64img(base["png"])
    k = base["px_per_cm"]
    anchors = base["anchors"]
    boxes = node_boxes(anchors)
    inkmask = ink(im)
    syms = {name: components(im, rgb) for name, rgb in COLORS.items()}
    notes = []
    new = []
    for a in arrows:
        a = dict(a)
        name = color_of(a["opts"])
        cands = syms.get(name) or []
        if not cands or a["anchor"] not in anchors:
            notes.append(f"{a['label']}: no {name} symbol found, left as is")
            new.append(a)
            continue
        # the symbol this arrow was pointing at: the one closest to its old start point
        ax, ay = anchors[a["anchor"]]
        old = (ax + a["sx"] * k, ay - a["sy"] * k)
        sym = min(cands, key=lambda s: (s["cx"] - old[0]) ** 2 + (s["cy"] - old[1]) ** 2)
        node = "b" if "b" in boxes and boxes["b"]["y0"] - 4 <= sym["cy"] <= boxes["b"]["y1"] + 4 else "a"
        nb = boxes[node]
        mid_y = (nb["y0"] + nb["y1"]) / 2
        direction = "up" if sym["cy"] < mid_y else "down"
        if path_blocked(inkmask, sym, nb, direction):
            other = "down" if direction == "up" else "up"
            if not path_blocked(inkmask, sym, nb, other):
                direction = other
        # start: just outside the symbol
        sx_px = sym["cx"]
        sy_px = sym["y0"] - GAP if direction == "up" else sym["y1"] + GAP
        anchor = f"{node}.north" if direction == "up" else f"{node}.south"
        ax, ay = anchors[anchor]
        a["anchor"] = anchor
        a["sx"], a["sy"] = round((sx_px - ax) / k, 2), round(-(sy_px - ay) / k, 2)
        a["_sym"] = sym; a["_dir"] = direction; a["_node"] = node; a["_nb"] = nb; a["_mid"] = (nb["x0"] + nb["x1"]) / 2
        a["_sy_px"] = sy_px
        new.append(a)

    def local_edge(a, ex_cm):
        """Lowest (or highest) ink in the band the arrow travels through, so labels hug the formula locally."""
        sym, nb, direction = a["_sym"], a["_nb"], a["_dir"]
        x0 = int(min(sym["cx"], sym["cx"] + ex_cm * k) - 0.15 * k); x1 = int(max(sym["cx"], sym["cx"] + ex_cm * k) + 0.15 * k)
        x0, x1 = max(x0, 0), min(x1, inkmask.shape[1])
        band = inkmask[int(nb["y0"]):int(nb["y1"]) + 1, x0:x1]
        rows = np.nonzero(band.any(axis=1))[0]
        if len(rows) == 0:
            return nb["y1"] if direction == "down" else nb["y0"]
        return int(nb["y0"]) + (rows.max() if direction == "down" else rows.min())
    # sides: within each direction, the left half of the symbols (by x) send their label left, the right half right;
    # a lone arrow near the middle goes straight up or down
    for direction in ("up", "down"):
        group = sorted([a for a in new if a.get("_dir") == direction], key=lambda a: a["_sym"]["cx"])
        n = len(group)
        for i, a in enumerate(group):
            off = a["_sym"]["cx"] - a["_mid"]
            if n % 2 == 1 and i == n // 2:
                # the middle symbol goes to the side whose nearest neighbor is farther away
                gap_l = a["_sym"]["cx"] - group[i - 1]["_sym"]["cx"] if i > 0 else 1e9
                gap_r = group[i + 1]["_sym"]["cx"] - a["_sym"]["cx"] if i + 1 < n else 1e9
                a["side"], a["ex"] = ("left", -OUT_CM) if gap_l >= gap_r else ("right", OUT_CM)
            elif i < n / 2:
                a["side"], a["ex"] = "left", -OUT_CM
            else:
                a["side"], a["ex"] = "right", OUT_CM
        # labels on the same side stack in rows: the outermost symbol gets the row nearest the formula,
        # each one further in goes one row further out, so their arrows never cross
        for side in ("left", "right", "above", "below"):
            same = [a for a in group if a["side"] == side]
            same.sort(key=lambda a: -abs(a["_sym"]["cx"] - a["_mid"]))
            if not same:
                continue
            # one baseline for the whole side, so stacked rows are really rows
            edges = [local_edge(a, a["ex"]) for a in same]
            shared = max(edges) if direction == "down" else min(edges)
            for r, a in enumerate(same):
                edge = shared if len(same) > 1 else edges[0]
                if len(same) == 2 and side in ("left", "right"):
                    # two labels on one side sit beside each other on the same row: the outer symbol reaches
                    # further out, the inner one stays close, so neither arrow crosses the other's label.
                    # Only when the longer label still fits on the page; otherwise they stack.
                    outer = same[0]
                    reach = outer["ex"] + (1.7 if outer["ex"] > 0 else -1.7)
                    tip_x = outer["_sym"]["cx"] + reach * k
                    est_w = 0.19 * k * len(re.sub(r"\\[a-zA-Z]+|[{}$]", "", outer["label"]))
                    fits = (tip_x + 0.3 * k + est_w <= PAGE_W_PX - MARGIN_PX) if reach > 0 else (tip_x - 0.3 * k - est_w >= MARGIN_PX)
                    if fits:
                        if r == 0:
                            a["ex"] = round(reach, 2)
                        r = 0
                if direction == "down":
                    ey = -((edge + (LABEL_CLEAR_CM + r * ROW_CM) * k) - a["_sy_px"]) / k
                    a["ey"] = round(min(ey, -0.3), 2)
                else:
                    ey = -((edge - (LABEL_CLEAR_CM + r * ROW_CM) * k) - a["_sy_px"]) / k
                    a["ey"] = round(max(ey, 0.3), 2)
        for a in group:
            a["ex"] = round(a["ex"], 2)
            if a["ex"] == 0:
                a["bend"] = 0
            else:
                a["bend"] = 15
                a["bdir"] = "left" if (direction == "down") == (a["ex"] < 0) else "right"
    return new, notes


def labels_only_tex(tex, arrows):
    """The block with the arrows' lines drawn in white, so a render shows only the labels (in their colors)."""
    hidden = []
    for a in arrows:
        h = dict(a)
        name = color_of(a["opts"]) or "black"
        h["opts"] = re.sub(r"\b" + name + r"\b", "white", a["opts"], count=1) if name != "black" else a["opts"] + ",white"
        h["nodeopts"] = (a.get("nodeopts") or "") + f",text={name}"
        hidden.append(h)
    return ft.apply_arrows(tex, hidden)


def label_boxes(tex, arrows, base):
    """Measure the rendered labels: one box per arrow, matched by color and proximity to the arrow tip."""
    png, err = ft.render(labels_only_tex(tex, arrows))
    if png is None:
        return None, err
    im = np.asarray(Image.open(io.BytesIO(png)).convert("RGB").crop(base["box"])).astype(int)
    k = base["px_per_cm"]
    boxes = []
    for a in arrows:
        name = color_of(a["opts"])
        ax, ay = base["anchors"].get(a["anchor"], (0, 0))
        tip = (ax + (a["sx"] + a["ex"]) * k, ay - (a["sy"] + a["ey"]) * k)
        cands = [c for c in components(im, COLORS[name], merge=8, min_px=4)] if name else []
        # the symbols are this color too, but they sit inside the formula; labels sit outside it
        nodes = node_boxes(base["anchors"]).values()
        outside = [c for c in cands if not any(nb["x0"] - 2 <= c["cx"] <= nb["x1"] + 2 and nb["y0"] - 2 <= c["cy"] <= nb["y1"] + 2 for nb in nodes)]
        pool = outside or cands
        boxes.append(min(pool, key=lambda c: (c["cx"] - tip[0]) ** 2 + (c["cy"] - tip[1]) ** 2) if pool else None)
    return boxes, None


def overlaps(p, q, pad=4):
    return not (p["x1"] + pad < q["x0"] or q["x1"] + pad < p["x0"] or p["y1"] + pad < q["y0"] or q["y1"] + pad < p["y0"])


def resolve(block, base, arrows, rounds=4):
    """Push labels apart / inside the text area, re-measuring after every change."""
    k = base["px_per_cm"]
    inkmask = ndimage.binary_dilation(ink(b64img(base["png"])), iterations=3)
    notes = []
    for _ in range(rounds):
        boxes, err = label_boxes(block["tex"], arrows, base)
        if boxes is None:
            notes.append("labels-only render failed: " + (err or "")[:120]); break
        changed = False
        for i, (a, bx) in enumerate(zip(arrows, boxes)):
            if bx is None:
                ax, ay = base["anchors"].get(a["anchor"], (0, 0))
                tip_x = ax + (a["sx"] + a["ex"]) * k
                if (tip_x > PAGE_W_PX - MARGIN_PX and a["ex"] > 0) or (tip_x < MARGIN_PX and a["ex"] < 0):
                    a["ex"], a["bend"] = 0.0, 0
                    a["side"] = "below" if a["ey"] < 0 else "above"
                    changed = True
                continue
            # off the text area: hang the label straight under (or over) its symbol instead
            if (bx["x0"] < MARGIN_PX and a["ex"] < 0) or (bx["x1"] > PAGE_W_PX - MARGIN_PX and a["ex"] > 0):
                a["ex"], a["bend"] = 0.0, 0
                a["side"] = "below" if a["ey"] < 0 else "above"
                changed = True; continue
            # over the formula's ink: push further out
            region = inkmask[max(bx["y0"], 0):bx["y1"] + 1, max(bx["x0"], 0):bx["x1"] + 1]
            if region.size and region.any():
                a["ey"] = round(a["ey"] + (0.25 if a["ey"] > 0 else -0.25), 2); changed = True
            if changed:
                continue
            # into another label: stagger the one that is further from the formula's middle
            for j, (b, by) in enumerate(zip(arrows, boxes)):
                if j <= i or by is None or not overlaps(bx, by):
                    continue
                victim = a if abs(a["ex"]) >= abs(b["ex"]) else b
                step = (by["y1"] - by["y0"] + 6) / k
                victim["ey"] = round(victim["ey"] + (step if victim["ey"] > 0 else -step), 2)
                changed = True
                break
        if not changed:
            break
    return arrows, notes


RESERVE_RE = re.compile(r"[ \t]*\\path\[room for labels\][^\n]*\n")


def reserve_room(tex, arrows, base, slack_cm=0.15):
    """Make the block as tall as its labels: an invisible path that stretches the picture's bounding box
    (the arrows themselves are overlays and take no space). Only the overhang beyond `slack_cm` is reserved,
    since the block already has some spacing around it."""
    tex = RESERVE_RE.sub("", tex)
    boxes, err = label_boxes(tex, arrows, base)
    if not boxes or all(b is None for b in boxes):
        return tex
    nodes = node_boxes(base["anchors"])
    top = min(nb["y0"] for nb in nodes.values()); bottom = max(nb["y1"] for nb in nodes.values())
    k = base["px_per_cm"]
    over_top = max(0.0, (top - min(b["y0"] for b in boxes if b)) / k - slack_cm)
    over_bottom = max(0.0, (max(b["y1"] for b in boxes if b) - bottom) / k - slack_cm)
    if over_top < 0.05 and over_bottom < 0.05:
        return tex
    top_node = "a"; bottom_node = "b" if "b" in nodes else "a"
    line = f"\\path[room for labels] ([yshift={over_top:.2f}cm]{top_node}.north) ([yshift=-{over_bottom:.2f}cm]{bottom_node}.south);\n"
    m = re.search(r"\n([ \t]*)\\draw", tex)
    if m:
        return tex[: m.start() + 1] + m.group(1) + line + tex[m.start() + 1:]
    closing = re.search(r"\n[ \t]*\}\s*\n\\end\{center\}\s*$", tex)
    return tex[: closing.start() + 1] + "    " + line + tex[closing.start() + 1:]


def render_crop(tex, base):
    png, err = ft.render(tex)
    if png is None:
        return None
    return Image.open(io.BytesIO(png)).convert("RGB").crop(base["box"])


def sheet(pairs, path):
    """Before/after rows, one per formula."""
    rows = []
    for title, before, after in pairs:
        w = max(before.width, after.width); h = max(before.height, after.height)
        row = Image.new("RGB", (2 * w + 30, h + 30), "white")
        d = ImageDraw.Draw(row); d.text((6, 6), title + "   (left: before, right: after)", fill=(90, 90, 90))
        row.paste(before, (0, 28)); row.paste(after, (w + 30, 28))
        d.line([(w + 15, 28), (w + 15, h + 28)], fill=(200, 200, 200), width=2)
        rows.append(row)
    W = max(r.width for r in rows); H = sum(r.height for r in rows)
    out = Image.new("RGB", (W, H), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y)); y += r.height
    out.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chapter", help="chapter file, e.g. 3-classification.tex")
    ap.add_argument("--skip", default="", help="comma-separated section names to leave alone")
    ap.add_argument("--only", default="", help="comma-separated section names to process")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    ft.ensure_format()
    blocks = [b for b in ft.find_blocks() if b["file"] == args.chapter and b["section"] not in skip and (not only or b["section"] in only)]
    print(f"{len(blocks)} formulas in {args.chapter}" + (f" (skipping {', '.join(sorted(skip))})" if skip else ""))
    pairs, results = [], []
    for b in blocks:
        base = ft.base_for(b["tex"])
        if "error" in base:
            print(f"  {b['section']}: base render failed"); continue
        arrows = ft.parse_arrows(b["tex"])
        new, notes = plan(b, base, arrows)
        new, notes2 = resolve(b, base, new)
        for a in new:
            for key in ("_sym", "_dir", "_node", "_mid", "_nb", "_sy_px"):
                a.pop(key, None)
        new_tex = reserve_room(ft.apply_arrows(b["tex"], new), new, base)
        before, after = render_crop(b["tex"], base), render_crop(new_tex, base)
        ok = after is not None
        print(f"  {b['section']}: {len(arrows)} arrows, {'ok' if ok else 'RENDER FAILED'}" + ("; " + "; ".join(notes + notes2) if notes + notes2 else ""))
        if before is not None and after is not None:
            pairs.append((f"{b['section']}", before, after))
        results.append((b, new_tex, ok))
    out = Path(args.out or "/private/tmp/claude-501/-Users-santiago-NannyML-The-Little-Book-of-ML-Metrics/f8838707-5f20-4259-8ba9-fa4b5a689b34/scratchpad") / f"layout_{args.chapter.replace('.tex', '')}.png"
    if pairs:
        print("sheet:", sheet(pairs, out))
    if args.write:
        for b, new_tex, ok in results:
            if not ok:
                continue
            path = ft.BOOK / b["file"]
            text = path.read_text()
            cur = next((x for x in ft.find_blocks() if x["file"] == b["file"] and x["section"] == b["section"] and x["start"] == b["start"]), None)
            if cur is None:
                # offsets shift as earlier blocks in the file change; find by section
                cur = next((x for x in ft.find_blocks() if x["file"] == b["file"] and x["tex"] == b["tex"]), None)
            if cur is None:
                print(f"  {b['section']}: could not relocate block, not written"); continue
            text = path.read_text()
            if not path.with_suffix(".tex.bak").exists():
                shutil.copy(path, path.with_suffix(".tex.bak"))
            path.write_text(text[: cur["start"]] + new_tex + text[cur["end"]:])
        print("written")


if __name__ == "__main__":
    main()
