"""Rebuild included book figures at hardcover print size.

Selects the approved generator for each asset. Writes candidates and numerical
preservation records under ignored review/; pass --install after reviewing them.
Run: uv run python tools/rebuild_figures.py [--source NAME] [--resume] [--install]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'review/hardcover-plots'
NOTEBOOKS = ROOT / 'notebooks'
MANIFEST = ROOT / 'review/hardcover-plots/manifest.json'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/book-inter-rollout-mpl')
os.environ.setdefault('MPLBACKEND', 'Agg')


def flatten_png(path):
    """Export opaque white-backed artwork; KDP requires flattened objects."""
    from PIL import Image
    with Image.open(path) as im:
        if im.mode == 'RGB': return
        rgba=im.convert('RGBA')
        flat=Image.new('RGB',rgba.size,'white')
        flat.paste(rgba,mask=rgba.getchannel('A'))
        flat.save(path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    return json.loads((ROOT / 'tools/figure-sources.json').read_text())


def worker(source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from PIL import Image
    import numpy as np
    sys.path.insert(0,str(NOTEBOOKS))
    import style
    manifest = json.loads(MANIFEST.read_text())
    wanted = {name for name,entry in manifest['figures'].items() if entry['source']==source}
    records = {}
    original_savefig = matplotlib.figure.Figure.savefig
    # Run each original generator in the original DejaVu configuration, so its
    # own tight_layout computes the previous axes geometry before substitution.
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({'axes.labelpad':15,'font.size':16,'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'})
    plt.show = lambda *args,**kwargs: None

    def payload(fig):
        data=[]
        for ax in fig.axes:
            item={'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'lines':[],'images':[],'collections':[]}
            if hasattr(ax,'get_zlim'): item['zlim']=list(ax.get_zlim())
            for line in ax.lines:
                item['lines'].append([np.asarray(line.get_xdata()).tolist(),np.asarray(line.get_ydata()).tolist(),line.get_color()])
            for im in ax.images:
                arr=np.asarray(im.get_array());item['images'].append({'shape':arr.shape,'sha':hashlib.sha256(arr.tobytes()).hexdigest(),'clim':im.get_clim(),'cmap':im.get_cmap().name})
            for coll in ax.collections:
                arr=coll.get_array()
                item['collections'].append({'array':None if arr is None else hashlib.sha256(np.asarray(arr).tobytes()).hexdigest(),'clim':coll.get_clim(),'cmap':coll.get_cmap().name})
            data.append(item)
        return hashlib.sha256(json.dumps(data,sort_keys=True,default=str).encode()).hexdigest()

    def capture(fig,name,*,dpi=300,**kw):
        name=Path(name).name
        if not name.endswith('.png'):name += '.png'
        if name not in wanted:return
        original=DEST/'original'/name
        if not original.exists():original=ROOT/'book/figures'/name
        baseline=DEST/'regenerated-original'/name
        output=DEST/'inter'/name
        for p in [baseline,output]:p.parent.mkdir(parents=True,exist_ok=True)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        positions=[ax.get_position().frozen() for ax in fig.axes]
        fig.set_layout_engine('none')
        for ax,position in zip(fig.axes,positions):ax.set_position(position)
        original_savefig(fig,baseline,dpi=dpi,bbox_inches='tight',pad_inches=0)
        before_data=payload(fig)
        before_text=[t.get_text() for t in fig.findobj(match=Text)]
        with plt.rc_context(style.PLOT_FONT_SETTINGS):
            style.prepare_book_figure(fig,name,manifest['figures'][name]['width_fraction'])
            fig.canvas.draw()
            crop=fig.get_tightbbox(fig.canvas.get_renderer()).frozen().padded(2/dpi)
            original_savefig(fig,output,dpi=dpi,bbox_inches=crop,pad_inches=0)
            flatten_png(output)
        if name!='BERTScore_matching.png':
            assert before_data==payload(fig),f'Data/scales changed: {name}'
        # Text may wrap; duplicate z-axis labels are intentionally removed.
        with Image.open(original) as im: orig_arr=np.asarray(im.convert('RGBA')); original_size=im.size
        with Image.open(baseline) as im: base_arr=np.asarray(im.convert('RGBA')); base_size=im.size
        with Image.open(output) as im: output_size=im.size
        pixel_equal=bool(orig_arr.shape==base_arr.shape and np.array_equal(orig_arr,base_arr))
        sizes=sorted(set(t.get_fontsize() for t in fig.findobj(match=Text) if t.get_visible() and t.get_text()))
        width_fraction=manifest['figures'][name]['width_fraction']
        printed=None if width_fraction is None else min(sizes)*style.PRINT_WIDTH_IN*72*width_fraction/(72*output_size[0]/dpi)
        from matplotlib.mathtext import MathTextParser
        parser = MathTextParser('path')
        glyph_sizes = []
        for t in fig.findobj(match=Text):
            if not t.get_visible() or not t.get_text(): continue
            if '$' in t.get_text():
                for line in t.get_text().split('\n'):
                    glyph_sizes.extend(g[1] for g in parser.parse(line, dpi=72, prop=t.get_fontproperties()).glyphs)
            else: glyph_sizes.append(t.get_fontsize())
        glyph_printed = min(glyph_sizes) * style.PRINT_WIDTH_IN * width_fraction / (output_size[0]/dpi)
        assert glyph_printed >= 7.0, (name, glyph_printed)
        records[name]={
            'smallest_printed_glyph_pt': glyph_printed,
            'source':source,'source_sha256':sha(NOTEBOOKS/source),
            'original_pixels':original_size,'regenerated_original_pixels':base_size,'inter_pixels':output_size,
            'original_pixels_match_replay':pixel_equal,
            'relative_aspect_change':(output_size[0]/output_size[1])/(original_size[0]/original_size[1])-1,
            'data_and_scale_sha256':before_data,'text_sha256':hashlib.sha256(json.dumps(before_text).encode()).hexdigest(),
            'font_sizes_pt':sizes,'smallest_printed_pt':printed,'target_printed_pt':7.2 if __import__('print_layout').dense_figure(Path(name).stem) else 8.0,'axes_positions':[list(p.bounds) for p in positions],
            'output_sha256':sha(output),'output':str(output),
        }
        print(f'{name}: Inter, replay pixel match={pixel_equal}',flush=True)

    style.save_figure=capture
    matplotlib.figure.Figure.savefig=lambda fig,path,*args,**kw: capture(fig,path,dpi=kw.get('dpi',300))
    os.chdir(NOTEBOOKS)
    try:
        if source.endswith('.ipynb'):
            cells=json.loads((NOTEBOOKS/source).read_text())['cells']
            if source.startswith('regression'):
                # The published MAPE/sMAPE images use the notebook's original
                # 10–50 grid. Re-run cell 1 after the independent MSLE grid.
                selects=[0,1,3,7,9,11,13,14,1,22,25,27,29]
                outputs={7:'MAE_3d_surface',11:'MSE_3d_surface',14:'MSLE_3d_surface',25:'MAPE_3d_surface',29:'sMAPE_3d_surface'}
            else:
                selects=[0,3,4,6,7,9,10]
                outputs={}
            ns={}
            for index in selects:
                code=''.join(cells[index]['source'])
                exec(compile(code,f'{source}:cell-{index}','exec'),ns)
                if index in outputs:capture(ns['fig'],outputs[index])
        elif source=='observability_plots.py':
            ns=runpy.run_path(str(NOTEBOOKS/source),run_name='plot_typography')
            ns['main'].__globals__['OUT']=DEST/'observability-data'
            ns['main'].__globals__['OUT'].mkdir(exist_ok=True)
            ns['main']()
        else:
            sys.argv=[str(NOTEBOOKS/source)]
            runpy.run_path(str(NOTEBOOKS/source),run_name='__main__')
    finally:
        (DEST/'records').mkdir(exist_ok=True)
        (DEST/'records'/f'{source}.json').write_text(json.dumps(records,indent=2)+'\n')
        plt.close('all')
    missing=wanted-records.keys()
    if missing:raise RuntimeError(f'Missing outputs from {source}: {sorted(missing)}')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source');parser.add_argument('--resume',action='store_true');parser.add_argument('--install',action='store_true')
    args=parser.parse_args()
    if args.source:
        if not MANIFEST.exists():
            DEST.mkdir(parents=True,exist_ok=True)
            MANIFEST.write_text(json.dumps({'figures':inventory()}))
        return worker(args.source)
    DEST.mkdir(parents=True,exist_ok=True)
    if args.resume:manifest=json.loads(MANIFEST.read_text())
    else:
        manifest={'baseline_commit':'6340c7529d6938dee200aad873a79a68dcfe182d','font_family':'Inter','emphasis_weight':600,'figures':inventory()}
        MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
    errors={}
    from concurrent.futures import ThreadPoolExecutor
    sources = sorted(set(e['source'] for e in manifest['figures'].values())-{None})
    def run_source(source):
        if args.resume and (DEST/'records'/f'{source}.json').exists():
            records = json.loads((DEST/'records'/f'{source}.json').read_text())
            wanted = {n for n,e in manifest['figures'].items() if e['source']==source}
            if wanted <= records.keys() and all('smallest_printed_glyph_pt' in records[n] for n in wanted): return None
        log=DEST/f'{source}.log'
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--source',source],stdout=stream,stderr=subprocess.STDOUT)
        print(source,result.returncode,flush=True)
        return (source,str(log)) if result.returncode else None
    with ThreadPoolExecutor(max_workers=2) as pool:
        for failure in pool.map(run_source,sources):
            if failure: errors[failure[0]]=failure[1]
    manifest['errors']=errors
    for record in (DEST/'records').glob('*.json'):
        rows=json.loads(record.read_text())
        for name,entry in rows.items():
            flatten_png(Path(entry['output']))
            entry['output_sha256']=sha(Path(entry['output']))
            if manifest['figures'][name]['source']==entry['source']:
                manifest['figures'][name].update(entry)
        record.write_text(json.dumps(rows,indent=2)+'\n')
    MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Generated',sum('output' in e for e in manifest['figures'].values()),'of',len(manifest['figures']))
    if errors:raise SystemExit(1)
    if args.install:
        import shutil
        for name, entry in manifest['figures'].items():
            shutil.copy2(entry['output'], ROOT / 'book/figures' / name)

if __name__=='__main__':main()
