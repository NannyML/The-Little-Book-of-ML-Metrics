"""Build and preflight the 5.5 x 8.5 inch KDP hardcover interior.

Run from the repository root:
    uv run --extra print python tools/hardcover.py
Use --no-build to preflight/export an already compiled review/hardcover/main.pdf.
The report is local and ignored. A passing report still needs KDP Previewer and
an actual printed proof; it is not an approval by Amazon.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, NameObject, RectangleObject

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / 'review/hardcover/main.pdf'
OUT = ROOT / 'output/pdf/the-little-book-of-ml-metrics-kdp-hardcover.pdf'
MM = 25.4 / 72
SOURCES = [
 'https://kdp.amazon.com/en_US/help/topic/GVBQ3CMEQW3W2VL6',
 'https://kdp.amazon.com/en_US/help/topic/G202145060',
 'https://kdp.amazon.com/en_US/help/topic/G201834260',
 'https://kdp.amazon.com/en_US/help/topic/GKYZRXFBZH2LDXAK',
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def page_mapping(directory):
    result = {}
    for path in directory.glob('*.aux'):
        for label, section, page in re.findall(r'\\newlabel\{(sec:[^}]+)\}\{\{([^}]*)\}\{(\d+)\}',path.read_text()):
            result[label] = {'section':section, 'page':int(page)}
    return result

def resources_check(reader):
    fonts = {}; transparency = []; layers = []
    visited=set()
    def walk(res, page):
        res = res.get_object()
        for f in res.get('/Font',{}).values():
            f=f.get_object(); name=str(f.get('/BaseFont'))
            for descendant in f.get('/DescendantFonts',[f]):
                d=descendant.get_object().get('/FontDescriptor',{}).get_object()
                fonts[name]=any(k in d for k in ('/FontFile','/FontFile2','/FontFile3'))
        for gs in res.get('/ExtGState',{}).values():
            gs=gs.get_object()
            if gs.get('/ca',1)!=1 or gs.get('/CA',1)!=1 or gs.get('/SMask','/None')!='/None':
                transparency.append(page)
        if '/Properties' in res:layers.append(page)
        for obj in res.get('/XObject',{}).values():
            if obj.idnum in visited:continue
            visited.add(obj.idnum);obj=obj.get_object()
            if '/SMask' in obj or '/Mask' in obj:transparency.append(page)
            if '/Resources' in obj:walk(obj['/Resources'],page)
    for n,p in enumerate(reader.pages,1):walk(p['/Resources'],n)
    return fonts, sorted(set(transparency)), sorted(set(layers))

def printable_rules(page, reader):
    """Raise thin native rules (including TeX radical bars) after fitting.

    Track the graphics matrix, so 0.8 pt is the physical output width even
    inside a scaled equation. Glyph outlines and filled shapes are untouched.
    """
    stream = ContentStream(page.get_contents(), reader)
    matrix = (1., 0., 0., 1.)
    width = 1.
    stack = []
    output = []
    changed = 0
    for args, op in stream.operations:
        if op == b'q': stack.append((matrix, width))
        elif op == b'Q': matrix, width = stack.pop()
        elif op == b'w': width = float(args[0])
        elif op == b'cm':
            a,b,c,d = matrix
            e,f,g,h = map(float, args[:4])
            matrix = (a*e+c*f, b*e+d*f, a*g+c*h, b*g+d*h)
        if op in (b'S', b's', b'B', b'B*', b'b', b'b*'):
            a,b,c,d = matrix
            total = a*a+b*b+c*c+d*d
            det = abs(a*d-b*c)
            largest = math.sqrt((total+math.sqrt(max(0.,total*total-4*det*det)))/2)
            scale = det/largest if largest else 0.
            if scale and width*scale < .75:
                output.extend([([FloatObject(.8/scale)],b'w'),(args,op),([FloatObject(width)],b'w')])
                changed += 1
                continue
        output.append((args,op))
    stream.operations = output
    page.replace_contents(stream)
    return changed

def export():
    reader=PdfReader(PROOF); writer=PdfWriter()
    for page in reader.pages:
        for key in ['/Annots','/AA','/Metadata']: page.pop(NameObject(key),None)
        added=writer.add_page(page)
        printable_rules(added, writer)
    # Supply the final even page explicitly, rather than relying on KDP padding.
    if len(writer.pages)%2:writer.add_blank_page(width=405,height=630)
    for i,page in enumerate(writer.pages,1):
        x=0 if i%2 else 9
        page.trimbox=RectangleObject([x,9,x+396,621])
        page.bleedbox=RectangleObject([0,0,405,630])
        page.cropbox=RectangleObject([0,0,405,630])
    writer.metadata=None
    for key in ['/Metadata','/Outlines','/AcroForm','/Names','/OpenAction','/AA','/PageLabels','/OCProperties']:
        writer.root_object.pop(NameObject(key),None)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('wb') as f:writer.write(f)

def verify():
    errors=[]; reader=PdfReader(OUT); count=len(reader.pages)
    assert 75<=count<=550 and count%2==0
    gutter = 27 if count<=150 else 36 if count<=300 else 45 if count<=500 else 54
    fonts,transparency,layers=resources_check(reader)
    if not all(fonts.values()):errors.append('Unembedded fonts')
    if transparency:errors.append(f'Transparency on pages {transparency}')
    if layers:errors.append(f'Optional content on pages {layers}')
    assert not reader.is_encrypted and OUT.stat().st_size<650_000_000
    assert not reader.outline and not reader.metadata
    missing=[]; small=[]; blanks=[]; images=[]; folios=[]; band_pages=[]
    min_edges=[999.]*4; boxes=[]; rules=[]
    with pdfplumber.open(OUT) as pdf:
        for n,page in enumerate(pdf.pages,1):
            x=0 if n%2 else 9
            assert list(reader.pages[n-1].mediabox)==[0,0,405,630]
            assert list(reader.pages[n-1].trimbox)==[x,9,x+396,621]
            assert not reader.pages[n-1].get('/Annots')
            chars=[c for c in page.chars if c['text'].strip()]
            if not chars and not page.images and not page.rects:blanks.append(n)
            if any(not c['upright'] for c in chars):band_pages.append(n)
            for c in chars:
                left,right=c['x0']-x,x+396-c['x1']
                inner,outer=(left,right) if n%2 else (right,left)
                top,bottom=c['top']-9,621-c['bottom']
                edges=[inner,outer,top,bottom]
                min_edges=[min(a,b) for a,b in zip(min_edges,edges)]
                if inner<gutter-.05 or min(outer,top,bottom)<18-.05:
                    missing.append({'page':n,'text':c['text'],'edges_pt':edges})
                # Decorative list dots are not text; rotated labels have a
                # transformed bounding-box 'size', not the font's point size.
                if c['upright'] and c['text']!='•' and c['size']<7-.005:
                    small.append({'page':n,'text':c['text'],'size':c['size']})
            for im in page.images:
                images.append({'page':n,'pixels':list(im['srcsize']),'width_pt':im['width'],
                               'dpi':min(im['srcsize'][0]*72/im['width'],im['srcsize'][1]*72/im['height'])})
            page_folio=[c for c in chars if c['upright'] and c['top']>580]
            if page_folio:
                digits=''.join(c['text'] for c in sorted(page_folio,key=lambda c:c['x0']))
                assert digits==str(n),(n,digits)
                folios.append({'page':n,'bottom_mm':min(621-c['bottom'] for c in page_folio)*MM})
            # Cards and callouts: inspect their complete filled boundaries.
            for shape in page.curves+page.rects:
                if shape.get('fill') and shape.get('non_stroking_color') in [(0.92941,0.96863,0.92549),(0.98431,0.92549,0.92549),(1,0.96078,0.84314)]:
                    left,right=shape['x0']-x,x+396-shape['x1']
                    inside,outside=(left,right) if n%2 else (right,left)
                    boxes.append({'page':n,'inside_mm':inside*MM,'outside_mm':outside*MM})
                    if inside<gutter-.1 or outside<18-.1:errors.append(f'Unsafe panel on page {n}')
            rules.extend({'page':n,'width':r['linewidth']} for r in page.lines+page.curves
                         if r.get('stroke') and r.get('linewidth',0)>0)
    thin_rules=[r for r in rules if r['width']<.749]
    if thin_rules:errors.append('Native rule under 0.75pt')
    if missing:errors.append('Text outside safe margins')
    if small:errors.append('Text under 7 PDF points')
    if min(i['dpi'] for i in images)<300:errors.append('Image under 300 DPI')
    # Observe the actual shipped pages, not the page counter before a page break.
    log=(PROOF.parent/'main.log').read_text()
    spread_checks=re.findall(r'SPREAD(FIRST|END)\|([\d.]+)\|(\d+)\|(\d+)',log)
    assert len(spread_checks)==188, len(spread_checks)
    for stage,section,start,after in spread_checks:
        start,after=int(start),int(after)
        if start%2 or after!=start+(1 if stage=='FIRST' else 2):errors.append(f'Broken metric spread {section} {stage}')
    for term in ['Overfull \\hbox','Overfull \\vbox','Missing character','undefined references','There were undefined']:
        if term in log:errors.append(term)
    chapter_checks=re.findall(r'CHAPTERFIT\|([0-9]+)\|([0-9]+)\|([0-9]+)',log)
    assert len(chapter_checks)==11, chapter_checks
    for chapter,start,after in chapter_checks:
        if int(after)!=int(start)+1:errors.append(f'Chapter opener {chapter} overflowed')
    mappings=page_mapping(PROOF.parent)
    monitors=[v['page'] for v in mappings.values() if v['section'].startswith('11.')]
    assert len(monitors)==13 and all(b-a==1 for a,b in zip(sorted(monitors),sorted(monitors)[1:]))
    longest_run=run=0
    for n in range(1,count+1):
        run=run+1 if n in blanks else 0;longest_run=max(longest_run,run)
    if longest_run>4:errors.append('Too many consecutive blank pages')
    manifest_path=ROOT/'review/hardcover-plots/manifest.json'
    plot_report={}
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['errors']:errors.append('Figure generation failures')
        for name,row in manifest['figures'].items():
            if row['output_sha256']!=sha(ROOT/'book/figures'/name):errors.append('Stale figure '+name)
        # Scale the generation-time glyph measurements to actual PDF inclusion
        # widths. A later margin change must not silently invalidate the report.
        glyphs=[]
        for im in images:
            candidates=[r for r in manifest['figures'].values()
                        if r['inter_pixels']==im['pixels']]
            if not candidates:
                errors.append(f"Unverified image on page {im['page']}")
                continue
            glyphs.append(min(r['smallest_printed_glyph_pt'] * im['width_pt'] /
                              r.get('printed_width_pt',109.7/MM*r['width_fraction'])
                              for r in candidates))
        glyph=min(glyphs,default=0)
        if glyph<7:errors.append('Figure glyph smaller than 7pt')
        plot_report={'count':len(manifest['figures']),'minimum_glyph_pt':glyph,
                     'actual_inclusion_widths_checked':True,'data_and_scales_preserved':True}
    else:
        errors.append('Missing figure generation/preflight manifest; run tools/rebuild_figures.py')
    report={'passed':not errors,'errors':errors,'pdf':str(OUT),'sha256':sha(OUT),'pages':count,
     'trim_mm':[139.7,215.9],'pdf_mm':[142.875,222.25],'bleed_mm':3.175,
     'minimum_text_clearances_mm':dict(zip(['inside','outside','top','bottom'],min_edges)),
     'minimum_panel_clearance_mm':{'inside':min([b['inside_mm'] for b in boxes],default=None),'outside':min([b['outside_mm'] for b in boxes],default=None)},
     'minimum_native_rule_pt':min(r['width'] for r in rules),'thin_rules':thin_rules,
     'panel_count':len(boxes),'text_margin_errors':missing,'small_text':small,
     'embedded_fonts':fonts,'transparency_pages':transparency,'layer_pages':layers,
     'image_count':len(images),'minimum_image_dpi':min(i['dpi'] for i in images),
     'minimum_folio_bottom_mm':min(f['bottom_mm'] for f in folios),
     'chapter_openers':len(chapter_checks),'metric_spreads':94,'monitor_pages':monitors,'blank_pages':blanks,'longest_blank_run':longest_run,
     'plot_verification':plot_report,'page_mapping':mappings,'official_sources':SOURCES,
     'limitations':['KDP Previewer and a physical proof have not been run.',
      'The supplied 250-page cover template must be regenerated for the final page count.',
      'Layout/numerical preservation checks are not a new scientific audit of all metrics.']}
    # Convert distances once, at the reporting boundary.
    report['minimum_text_clearances_mm']={k:v*MM for k,v in report['minimum_text_clearances_mm'].items()}
    dest=ROOT/'design/hardcover/preflight.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','errors','pages','trim_mm','minimum_text_clearances_mm','minimum_panel_clearance_mm','minimum_image_dpi','plot_verification']},indent=2))
    if errors:raise SystemExit(1)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-build',action='store_true');args=parser.parse_args()
    if not args.no_build:
        PROOF.parent.mkdir(parents=True,exist_ok=True)
        with (PROOF.parent/'build.log').open('w') as f:
            subprocess.run(['latexmk','-xelatex','-interaction=nonstopmode','-halt-on-error','-outdir='+str(PROOF.parent),'main.tex'],cwd=ROOT/'book',stdout=f,stderr=subprocess.STDOUT,check=True)
    export();verify()

if __name__=='__main__':main()
