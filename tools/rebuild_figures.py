"""Rebuild the book's figures and write the hardcover figure manifest.

Each figure is drawn at its printed size by the generator named in
tools/figure-sources.json and saved straight into book/figures. While the
generators run, notebooks/style.py logs every saved figure's printed width and
smallest lettering; this script checks that every included figure was produced
and writes review/hardcover-plots/manifest.json for tools/hardcover.py.
Run: uv run python tools/rebuild_figures.py [--source GENERATOR.py ...]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks'
FIGURES = ROOT / 'book/figures'
SOURCES = ROOT / 'tools/figure-sources.json'
MANIFEST = ROOT / 'review/hardcover-plots/manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included():
    """Figure files included by the chapters, with their inclusion width."""
    found = {}
    for tex in sorted((ROOT / 'book').glob('*.tex')):
        for width, name in re.findall(r'includegraphics\[width=([.\d]*)\\textwidth\]\{figures/([^}]+\.png)\}',
                                      tex.read_text()):
            found[name] = {'chapter': tex.name, 'width_fraction': float(width or 1)}
    return found


def run(source, record):
    env = dict(os.environ, BOOK_FIGURE_RECORD=str(record), MPLBACKEND='Agg')
    result = subprocess.run([sys.executable, str(NOTEBOOKS / source)], cwd=NOTEBOOKS, env=env,
                            capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-2000:], result.stderr[-2000:], sep='\n')
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', nargs='*', help='only rerun these generators')
    args = parser.parse_args()
    sources = json.loads(SOURCES.read_text())
    figures = included()
    unknown = sorted(set(figures) - set(sources))
    if unknown:
        raise SystemExit(f'No generator listed in {SOURCES.name} for: {unknown}')
    wanted = sorted({sources[name]['source'] for name in figures})
    if args.source:
        wanted = [s for s in wanted if s in args.source]
    previous = json.loads(MANIFEST.read_text())['figures'] if MANIFEST.exists() else {}
    errors = {}
    with tempfile.TemporaryDirectory() as tmp:
        record = Path(tmp) / 'figures.jsonl'
        record.touch()
        for source in wanted:
            ok = run(source, record)
            print(f'{source}: {"ok" if ok else "FAILED"}', flush=True)
            if not ok:
                errors[source] = 'generator failed'
        rows = {}
        for line in record.read_text().splitlines():
            row = json.loads(line)
            rows[row['name']] = row
    manifest = {'figures': {}, 'errors': errors}
    for name, entry in sorted(figures.items()):
        source = sources[name]['source']
        row = rows.get(name) or (previous.get(name) if source not in wanted else None)
        if row is None:
            errors[name] = f'not produced by {source}'
            continue
        from PIL import Image
        path = FIGURES / name
        with Image.open(path) as im:
            pixels = list(im.size)
        manifest['figures'][name] = dict(entry, source=source, output=str(path), output_sha256=sha(path),
                                         inter_pixels=pixels, printed_width_pt=row['printed_width_pt'],
                                         smallest_printed_glyph_pt=row['smallest_printed_glyph_pt'])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{len(manifest["figures"])} of {len(figures)} figures recorded in {MANIFEST.relative_to(ROOT)}')
    if errors:
        raise SystemExit(f'Errors: {errors}')


if __name__ == '__main__':
    main()
