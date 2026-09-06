"""Interactive tuner for the book's annotated formulas.

Every formula in the book is a TikZ block: a node holding the math and one
`\\draw` arrow per colored symbol.  Getting the arrow start, end, bend and label
right takes trial and error, so this tool does the loop for you:

    uv run python tools/formula_tuner.py            # then open http://localhost:8765

(It is also embedded in tools/review_recorder.py: "Tune arrows" on any page with a formula.)

Left: every formula block in book/*.tex, grouped by chapter.  Middle: the block
compiled on its own (same preamble and fonts as the book, rendered by XeLaTeX
and pdftoppm).  Right: one card per arrow with sliders for the start offset,
the end offset, the bend and the label side, plus the label text.

While a slider moves, the arrows are drawn instantly in the browser on top of a
render of the bare formula (the node anchors are located once per formula with a
calibration compile).  Half a second after the last change the exact XeLaTeX
render replaces the preview.  "Save to .tex" writes the block back into the
chapter file exactly where it came from; nothing else in the file is touched.

Speed: the book preamble (everything except the fonts, which XeTeX cannot store
in a format) is precompiled once per run into a format file, so an exact render
takes ~0.4 s instead of ~0.8 s.  If the format cannot be built the tool falls
back to full compiles.

Requirements: xelatex and pdftoppm on PATH (the book's own toolchain).
"""
import base64
import hashlib
import http.server
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"
PORT = 8765
if os.path.isdir("/Library/TeX/texbin") and "/Library/TeX/texbin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] += ":/Library/TeX/texbin"   # MacTeX, when the tool is started from an app that lacks the shell PATH

CHAPTER_FILES = sorted(BOOK.glob("[0-9]*-*.tex"), key=lambda p: int(p.name.split("-")[0]))

BLOCK_RE = re.compile(r"\\begin\{center\}\s*\n\s*\\tikz\{.*?\n\s*\}\s*\n\\end\{center\}", re.S)
SECTION_RE = re.compile(r"\\section\{((?:[^{}]|\{[^{}]*\})+)\}")
ARROW_RE = re.compile(
    r"\\draw\[(?P<opts>[^\]]*)\]\s*\(\$\((?P<anchor>[ab]\.[a-z ]+)\)\s*\+\s*\((?P<sx>-?[\d.]+)\s*,\s*(?P<sy>-?[\d.]+)\)\$\)\s*"
    r"to(?:\[bend (?P<bdir>left|right)\s*=\s*(?P<bend>-?\d+)\])?\s*node\[pos=1,\s*(?P<side>left|right|above|below)(?P<nodeopts>[^\]]*)\]\s*\{(?P<label>.*?)\}\s*"
    r"\+\((?P<ex>-?[\d.]+)\s*,\s*(?P<ey>-?[\d.]+)\)\s*;",
    re.S,
)
FONT_RE = re.compile(r"\\(?:setmainfont|setsansfont|setmonofont|newfontfamily\\?\w*)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}\s*(?:\[[^\]]*\])?", re.S)

DPI = 170
PX_PER_CM = DPI / 2.54
PAD_CM = 3.2                       # vertical room kept around the formula for arrows and labels
ANCHORS = ["north", "south", "east", "west", "north east", "north west", "south east", "south west", "center"]
NML_COLORS = ["nmlred", "nmlcyan", "nmlpurple", "nmlgreen", "nmlyellow"]


def preamble():
    main = (BOOK / "main.tex").read_text()
    return main[: main.index("\\begin{document}")]


def find_blocks():
    """Return a list of dicts describing every formula block in the book."""
    blocks = []
    for f in CHAPTER_FILES:
        text = f.read_text()
        sections = [(m.start(), m.group(1)) for m in SECTION_RE.finditer(text)]
        for m in BLOCK_RE.finditer(text):
            if "\\node[inner sep=2pt" not in m.group(0):
                continue
            sec = next((s for pos, s in reversed(sections) if pos < m.start()), "?")
            blocks.append({"id": f"{f.name}:{m.start()}", "file": f.name, "start": m.start(), "end": m.end(),
                           "section": sec, "tex": m.group(0)})
    return blocks


def parse_arrows(tex):
    arrows = []
    for m in ARROW_RE.finditer(tex):
        d = m.groupdict()
        arrows.append({
            "span": [m.start(), m.end()],
            "opts": d["opts"], "anchor": d["anchor"],
            "sx": float(d["sx"]), "sy": float(d["sy"]),
            "bdir": d["bdir"] or "left", "bend": int(d["bend"]) if d["bend"] else 0,
            "side": d["side"], "nodeopts": d["nodeopts"] or "", "label": d["label"],
            "ex": float(d["ex"]), "ey": float(d["ey"]),
        })
    return arrows


def arrow_tex(a):
    bend = f"to[bend {a['bdir']}={a['bend']}]" if int(a["bend"]) else "to"
    return (f"\\draw[{a['opts']}] ($({a['anchor']})+({a['sx']:.2f},{a['sy']:.2f})$) {bend} "
            f"node[pos=1, {a['side']}{a.get('nodeopts', '')}] {{{a['label']}}} +({a['ex']:.2f},{a['ey']:.2f});")


def apply_arrows(tex, arrows):
    """Rebuild the block text with the given arrow parameters (spans from the original)."""
    out = tex
    for a in sorted(arrows, key=lambda x: -x["span"][0]):
        out = out[: a["span"][0]] + arrow_tex(a) + out[a["span"][1]:]
    return out


def strip_arrows(tex):
    """The block without its arrows: the bare formula node(s)."""
    out = tex
    for a in sorted(parse_arrows(tex), key=lambda x: -x["span"][0]):
        out = out[: a["span"][0]] + out[a["span"][1]:]
    return re.sub(r"\n[ \t]*\n", "\n", out)


def changed_arrows(old, new):
    """Labels of arrows whose parameters differ between two block texts, plus 'block text' if anything else differs."""
    def vals(a):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items() if k != "span"}
    ao, an = parse_arrows(old), parse_arrows(new)
    changed = [an[i]["label"] for i in range(min(len(ao), len(an))) if vals(ao[i]) != vals(an[i])]
    if len(ao) != len(an) or re.sub(r"\s+", "", strip_arrows(old)) != re.sub(r"\s+", "", strip_arrows(new)):
        changed.append("block text")
    return changed


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------
RENDER_LOCK = threading.Lock()
WORK = Path(tempfile.mkdtemp(prefix="formula_tuner_"))
FMT = {"path": None, "hash": None, "fonts": "", "tried": None}


def _run_xelatex(args, name):
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={WORK}", *args],
                       cwd=BOOK, capture_output=True, text=True, timeout=180)
    pdf = WORK / f"{name}.pdf"
    if not pdf.exists():
        log = (WORK / f"{name}.log").read_text(errors="replace") if (WORK / f"{name}.log").exists() else r.stdout
        err = "\n".join(l for l in log.splitlines() if l.startswith("!") or l.startswith("l."))
        return None, err or log[-2000:]
    return pdf, None


def ensure_format():
    """Precompile the preamble (minus fontspec and the fonts) into WORK/tunerfmt.fmt. Safe to call often."""
    pre = preamble()
    h = hashlib.md5(pre.encode()).hexdigest()
    if FMT["tried"] == h:
        return
    FMT.update(tried=h, path=None)
    fonts = "\n".join(FONT_RE.findall(pre))
    body_pre = FONT_RE.sub("", pre)
    if "\\usepackage{fontspec}\n" not in body_pre:
        return
    body_pre = body_pre.replace("\\usepackage{fontspec}\n", "", 1)
    # keep every font in the dump a classic TFM font: XeTeX cannot dump native (OpenType) fonts
    body_pre = ("\\renewcommand{\\encodingdefault}{OT1}\\renewcommand{\\rmdefault}{cmr}"
                "\\renewcommand{\\sfdefault}{cmss}\\renewcommand{\\ttdefault}{cmtt}\n" + body_pre)
    (WORK / "tunerpre.tex").write_text(body_pre + "\n\\csname endofdump\\endcsname\n")
    with RENDER_LOCK:
        for old in WORK.glob("tunerfmt.*"):
            old.unlink()
        subprocess.run(["xelatex", "-ini", "-interaction=nonstopmode", "-jobname=tunerfmt", f"-output-directory={WORK}",
                        "&xelatex", "mylatexformat.ltx", str(WORK / "tunerpre.tex")],
                       cwd=BOOK, capture_output=True, text=True, timeout=300)
    fmt = WORK / "tunerfmt.fmt"
    if not fmt.exists():
        return
    FMT.update(path=fmt, hash=h, fonts=fonts)
    # smoke test: the fonts must load in the body and produce a page
    png, err = render("\\begin{center}\\tikz{\\node[inner sep=2pt, font=\\Large] (a) {$x$};}\\end{center}")
    if png is None:
        FMT["path"] = None


def render(tex, extra_preamble=""):
    """Compile one block with the book preamble; return (full-page PNG bytes, None) or (None, error)."""
    doc_body = f"{extra_preamble}\n\\begin{{document}}\n\\pagestyle{{empty}}\n\\vspace*{{2.2cm}}\n{tex}\n\\vspace*{{2.2cm}}\n\\end{{document}}\n"
    with RENDER_LOCK:
        for old in WORK.glob("f.*"):
            old.unlink()
        if FMT["path"]:
            # the first TeX command starts mylatexformat's scan; the \endofdump line ends the skipped part
            (WORK / "f.tex").write_text("\\relax\n\\endofdump\n\\usepackage{fontspec}\n" + FMT["fonts"] + "\n" + doc_body)
            pdf, err = _run_xelatex([f"-fmt={FMT['path']}", str(WORK / "f.tex")], "f")
            if pdf is None and "endofdump" in (err or ""):
                FMT["path"] = None      # format unusable: fall through to a full compile
        if not FMT["path"]:
            (WORK / "f.tex").write_text(preamble() + "\n" + doc_body)
            pdf, err = _run_xelatex([str(WORK / "f.tex")], "f")
        if pdf is None:
            return None, err
        subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-f", "1", "-l", "1", "-singlefile", str(pdf), str(WORK / "f")],
                       check=True, capture_output=True)
        return (WORK / "f.png").read_bytes(), None


# ---------------------------------------------------------------------------
# Base image + anchor calibration (per bare formula, cached)
# ---------------------------------------------------------------------------
BASE_CACHE = {}


def _mark_color(node, k):
    return (255, 0, 16 * k) if node == "a" else (0, 255, 16 * k)


def calibration_tex(bare):
    """The bare block with the node text in white and a colored 1.2pt square on every anchor."""
    nodes = ["a"] + (["b"] if re.search(r"\]\s*\(b\)\s*\{", bare) else [])
    marks = []
    for node in nodes:
        for k, anc in enumerate(ANCHORS):
            r, g, b = _mark_color(node, k)
            marks.append(f"\\fill[overlay, color={{rgb,255:red,{r};green,{g};blue,{b}}}] ({node}.{anc}) ++(-0.6pt,-0.6pt) rectangle ++(1.2pt,1.2pt);")
    closing = re.search(r"\n[ \t]*\}\s*\n\\end\{center\}\s*$", bare)
    tex = bare[: closing.start()] + "\n" + "\n".join(marks) + bare[closing.start():]
    tex = tex.replace("\\tikz{", "\\tikz[text=white]{", 1)
    return tex, nodes


def base_for(tex):
    """Render the bare formula once, locate its anchors, and fix the crop box used for every render of this block."""
    from PIL import Image
    import numpy as np
    bare = strip_arrows(tex)
    key = (bare, bool(FMT["path"]))
    if key in BASE_CACHE:
        return BASE_CACHE[key]
    png, err = render(bare)
    if png is None:
        return {"error": err}
    cal_tex, nodes = calibration_tex(bare)
    whites = "\n".join(f"\\definecolor{{{c}}}{{RGB}}{{255,255,255}}" for c in NML_COLORS)
    cal, err = render(cal_tex, extra_preamble=whites)
    if cal is None:
        return {"error": err}
    im = np.asarray(Image.open(io.BytesIO(cal)).convert("RGB")).astype(int)
    anchors = {}
    for node in nodes:
        for k, anc in enumerate(ANCHORS):
            r, g, b = _mark_color(node, k)
            hit = (abs(im[:, :, 0] - r) < 6) & (abs(im[:, :, 1] - g) < 6) & (abs(im[:, :, 2] - b) < 6)
            ys, xs = np.nonzero(hit)
            if len(xs):
                anchors[f"{node}.{anc}"] = [float(xs.mean()), float(ys.mean())]
    if not anchors:
        return {"error": "calibration failed: no anchor marks found"}
    H, W = im.shape[:2]
    ys = [v[1] for v in anchors.values()]
    pad = PAD_CM * PX_PER_CM
    box = (0, max(0, int(min(ys) - pad)), W, min(H, int(max(ys) + pad)))
    base = Image.open(io.BytesIO(png)).convert("RGB").crop(box)
    out = io.BytesIO(); base.save(out, format="PNG")
    entry = {"png": base64.b64encode(out.getvalue()).decode(), "box": box, "px_per_cm": PX_PER_CM,
             "anchors": {k: [v[0] - box[0], v[1] - box[1]] for k, v in anchors.items()}}
    BASE_CACHE[key] = entry
    return entry


def crop_to(png, box):
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB").crop(box)
    out = io.BytesIO(); im.save(out, format="PNG"); return out.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Formula tuner</title>
<style>
 @font-face{font-family:'RubikT';src:url('/fonts/Rubik-Regular.ttf')}
 @font-face{font-family:'RubikT';font-style:italic;src:url('/fonts/Rubik-Italic.ttf')}
 body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;display:grid;grid-template-columns:260px 1fr 380px;height:100vh;color:#222}
 body.embed{grid-template-columns:1fr 380px}
 body.embed #list{display:none}
 #list{overflow:auto;border-right:1px solid #ddd;padding:8px;font-size:13px}
 #list h4{margin:10px 0 4px;color:#666;font-weight:600;font-size:12px;text-transform:uppercase}
 #list div.item{padding:4px 6px;border-radius:4px;cursor:pointer}
 #list div.item:hover{background:#eef}
 #list div.item.sel{background:#0AA7D4;color:#fff}
 #view{display:flex;flex-direction:column;align-items:center;padding:12px;overflow:auto;background:#fafafa}
 #stage{position:relative;max-width:100%;border:1px solid #ddd;background:#fff}
 #stage img{display:block;max-width:100%}
 #ov{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
 #status{font-size:12px;color:#777;margin:6px}
 #status.live{color:#0AA7D4}
 pre.err{color:#b00;font-size:12px;white-space:pre-wrap}
 #ctrl{overflow:auto;border-left:1px solid #ddd;padding:10px;font-size:13px}
 .card{border:1px solid #e3e3e3;border-radius:6px;padding:8px;margin-bottom:10px}
 .card .lab{font-weight:600;margin-bottom:6px}
 .row{display:grid;grid-template-columns:70px 1fr 48px;align-items:center;gap:6px;margin:3px 0}
 .row input[type=range]{width:100%}
 .row input[type=text],select{font:inherit;padding:2px 4px}
 button{font:inherit;padding:6px 10px;border-radius:5px;border:1px solid #bbb;background:#fff;cursor:pointer;margin-right:6px}
 button.primary{background:#0AA7D4;color:#fff;border-color:#0AA7D4}
 textarea{width:100%;height:160px;font:12px/1.35 Menlo,monospace}
 .swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
 #saved .banner{background:#e6f7ea;border:1px solid #4EB046;color:#1f6b2a;border-radius:6px;padding:8px 10px;margin-top:8px;font-size:13px;line-height:1.4}
 #saved .banner.err{background:#fbecec;border-color:#DD4040;color:#8a1f1f}
</style></head><body>
<div id="list"></div>
<div id="view"><div id="status">Pick a formula on the left.</div><div id="stage"><img id="img" alt=""><svg id="ov"></svg></div><pre class="err" id="err"></pre></div>
<div id="ctrl"></div>
<script>
const COLORS={nmlred:'#DD4040',nmlcyan:'#00C8E5',nmlpurple:'#3B0280',nmlgreen:'#4EB046',nmlyellow:'#E1BC29'};
const DPIpt=170/72.27;
let blocks=[],cur=null,arrows=[],timer=null,base=null,seq=0,inflight=false,dirty=false;
const Q=new URLSearchParams(location.search),EMBED=Q.get('embed')==='1';if(EMBED)document.body.classList.add('embed');
async function load(){blocks=await (await fetch('/api/blocks')).json();const L=document.getElementById('list');let f='';
 for(const b of blocks){if(b.file!==f){f=b.file;const h=document.createElement('h4');h.textContent=f;L.appendChild(h);}
  const d=document.createElement('div');d.className='item';d.textContent=b.section;d.onclick=()=>select(b,d);d.dataset.id=b.id;L.appendChild(d);}
 const want=Q.get('block');if(want){const b=blocks.find(x=>x.id===want);if(b)select(b,document.querySelector(`[data-id="${b.id}"]`));}}
async function select(b,el){document.querySelectorAll('#list .item').forEach(x=>x.classList.remove('sel'));if(el)el.classList.add('sel');
 cur=JSON.parse(JSON.stringify(b));arrows=cur.arrows;base=null;buildControls();
 document.getElementById('status').textContent='preparing preview…';document.getElementById('err').textContent='';
 const r=await fetch('/api/base',{method:'POST',body:JSON.stringify({tex:cur.tex})});const j=await r.json();
 if(j.error){document.getElementById('err').textContent=j.error;return;}
 base=j;showLive();render();}
function color(opts){for(const k in COLORS){if(opts.includes(k))return COLORS[k];}return '#888';}
function slider(card,a,key,min,max,step,name){const r=document.createElement('div');r.className='row';
 r.innerHTML=`<span>${name}</span><input type="range" min="${min}" max="${max}" step="${step}" value="${a[key]}"><input type="text" value="${a[key]}">`;
 const [lab,rng,txt]=r.children;rng.oninput=()=>{a[key]=parseFloat(rng.value);txt.value=rng.value;changed();};
 txt.onchange=()=>{a[key]=parseFloat(txt.value);rng.value=txt.value;changed();};card.appendChild(r);}
function buildControls(){const C=document.getElementById('ctrl');C.innerHTML='';
 const top=document.createElement('div');top.innerHTML=`<div style="margin-bottom:8px"><b>${cur.section}</b> <span style="color:#777">${cur.file}</span></div>
 <button class="primary" id="save">Save to .tex</button><button id="rerender">Re-render</button><button id="reset">Reset</button>${EMBED?'<button id="close" style="float:right">Close</button>':''}<div id="saved" style="color:#2a7;font-size:12px;margin-top:4px"></div>`;C.appendChild(top);
 if(EMBED)top.querySelector('#close').onclick=()=>parent.postMessage('tuner-close','*');
 top.querySelector('#save').onclick=save;top.querySelector('#rerender').onclick=render;
 top.querySelector('#reset').onclick=()=>{const b=blocks.find(x=>x.id===cur.id);select(b,document.querySelector(`[data-id="${b.id}"]`));};
 arrows.forEach((a,i)=>{const card=document.createElement('div');card.className='card';
  card.innerHTML=`<div class="lab"><span class="swatch" style="background:${color(a.opts)}"></span>arrow ${i+1}: <input type="text" value="${a.label.replace(/"/g,'&quot;')}" style="width:190px"></div>`;
  card.querySelector('input').oninput=e=>{a.label=e.target.value;changed();};
  const anc=document.createElement('div');anc.className='row';anc.innerHTML=`<span>anchor</span><select><option>a.north</option><option>a.south</option><option>a.east</option><option>a.west</option><option>a.north east</option><option>a.north west</option><option>b.south</option><option>b.north</option></select><span></span>`;
  const sel=anc.querySelector('select');if(![...sel.options].some(o=>o.value===a.anchor)){const o=document.createElement('option');o.textContent=a.anchor;sel.appendChild(o);}sel.value=a.anchor;sel.onchange=()=>{a.anchor=sel.value;changed();};card.appendChild(anc);
  slider(card,a,'sx',-6,6,0.05,'start x');slider(card,a,'sy',-1.5,1.5,0.05,'start y');
  slider(card,a,'ex',-3,3,0.05,'end dx');slider(card,a,'ey',-1.5,1.5,0.05,'end dy');
  slider(card,a,'bend',0,60,1,'bend °');
  const side=document.createElement('div');side.className='row';side.innerHTML=`<span>bend / label</span><select><option value="left">bend left</option><option value="right">bend right</option></select><select><option>left</option><option>right</option><option>above</option><option>below</option></select>`;
  const [s1,s2]=side.querySelectorAll('select');s1.value=a.bdir;s2.value=a.side;s1.onchange=()=>{a.bdir=s1.value;changed();};s2.onchange=()=>{a.side=s2.value;changed();};card.appendChild(side);
  C.appendChild(card);});
 const raw=document.createElement('div');raw.className='card';raw.innerHTML='<div class="lab">Raw block (edits here win over the sliders)</div><textarea id="raw"></textarea><button id="applyraw">Apply raw</button>';
 raw.querySelector('#raw').value=cur.tex;raw.querySelector('#applyraw').onclick=()=>{cur.tex=document.getElementById('raw').value;fetch('/api/reparse',{method:'POST',body:cur.tex}).then(r=>r.json()).then(j=>{cur.arrows=j;arrows=j;select(cur,document.querySelector('#list .item.sel'));});};
 C.appendChild(raw);}
function changed(){showLive();clearTimeout(timer);timer=setTimeout(render,450);}
// ---- instant preview: base image + SVG arrows ----
function showLive(){if(!base)return;const img=document.getElementById('img');const src='data:image/png;base64,'+base.png;if(img.src!==src)img.src=src;
 drawOverlay();document.getElementById('ov').style.display='';const st=document.getElementById('status');st.textContent='preview (exact render follows)';st.className='live';}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function inlineSvg(text){const parts=text.split('$');let inner='';parts.forEach((p,i)=>{if(!p)return;inner+=i%2?`<tspan font-style="italic">${esc(p.replace(/\\[a-zA-Z]+|[{}^_]/g,''))}</tspan>`:esc(p.replace(/\\[a-zA-Z]+\{?|[{}]/g,''));});return inner;}
function labelSvg(text,x,y,side,col,sep,fs){const lines=text.split('\\\\'),n=lines.length,lh=1.2*fs;let anchor='middle',top;
 if(side==='left'){x-=sep;anchor='end';top=y-(n-1)*lh/2;}else if(side==='right'){x+=sep;anchor='start';top=y-(n-1)*lh/2;}
 else if(side==='above'){y-=sep;top=y-(n-1)*lh-0.3*fs;}else{y+=sep;top=y+0.55*fs;}
 return lines.map((l,i)=>`<text x="${x}" y="${top+i*lh}" dy="0.35em" text-anchor="${anchor}" font-family="RubikT,Rubik,sans-serif" font-size="${fs}" fill="${col}">${inlineSvg(l)}</text>`).join('');}
function drawOverlay(){const ov=document.getElementById('ov'),img=document.getElementById('img');if(!base)return;
 const W=img.naturalWidth||1,H=img.naturalHeight||1;ov.setAttribute('viewBox',`0 0 ${W} ${H}`);
 const k=base.px_per_cm,lw=0.6*DPIpt,sep=3.333*DPIpt,fs=10*DPIpt;let s='';
 for(const a of arrows){const anc=base.anchors[a.anchor];if(!anc)continue;const col=color(a.opts);
  const S={x:anc[0]+a.sx*k,y:anc[1]-a.sy*k},E={x:S.x+a.ex*k,y:S.y-a.ey*k};
  const dx=E.x-S.x,dy=-(E.y-S.y),th=Math.atan2(dy,dx),al=(a.bdir==='right'?-1:1)*(a.bend||0)*Math.PI/180,d=0.3915*Math.hypot(dx,dy);
  const o=th+al,n=th+Math.PI-al;const C1={x:S.x+d*Math.cos(o),y:S.y-d*Math.sin(o)},C2={x:E.x+d*Math.cos(n),y:E.y-d*Math.sin(n)};
  s+=`<path d="M${S.x},${S.y} C${C1.x},${C1.y} ${C2.x},${C2.y} ${E.x},${E.y}" fill="none" stroke="${col}" stroke-width="${lw}"/>`;
  let tx=E.x-C2.x,ty=E.y-C2.y,tl=Math.hypot(tx,ty)||1;tx/=tl;ty/=tl;const L=10*lw,w=3.5*lw;
  s+=`<path d="M${E.x},${E.y} L${E.x-L*tx-w*ty},${E.y-L*ty+w*tx} L${E.x-0.7*L*tx},${E.y-0.7*L*ty} L${E.x-L*tx+w*ty},${E.y-L*ty-w*tx} Z" fill="${col}"/>`;
  s+=labelSvg(a.label,E.x,E.y,a.side,col,sep,fs);}
 ov.innerHTML=s;}
document.getElementById('img').onload=()=>{if(document.getElementById('ov').style.display!=='none')drawOverlay();};
// ---- exact render, coalesced ----
async function render(){if(!cur||!base)return;if(inflight){dirty=true;return;}inflight=true;const my=++seq;
 const st=document.getElementById('status');st.textContent=(st.className==='live'?'preview · ':'')+'rendering…';document.getElementById('err').textContent='';
 try{const r=await fetch('/api/render',{method:'POST',body:JSON.stringify({tex:cur.tex,arrows:arrows})});const j=await r.json();
  if(my===seq&&!dirty){if(j.png){document.getElementById('ov').style.display='none';document.getElementById('img').src='data:image/png;base64,'+j.png;st.textContent=`exact render, ${j.ms} ms`;st.className='';document.getElementById('raw').value=j.tex;}
   else{st.textContent='compile error';st.className='';document.getElementById('err').textContent=j.error;}}}
 finally{inflight=false;if(dirty){dirty=false;render();}}}
async function save(){const r=await fetch('/api/save',{method:'POST',body:JSON.stringify({id:cur.id,tex:cur.tex,arrows:arrows})});const j=await r.json();
 const S=document.getElementById('saved');
 if(j.ok){const what=j.changed.length?`${j.changed.length} arrow${j.changed.length>1?'s':''} changed: ${j.changed.map(l=>'“'+l+'”').join(', ')}`:'nothing changed since the last save';
  S.innerHTML=`<div class="banner">&#10003; Saved to <b>${j.file}</b> (${j.section}) at ${j.time}.<br>${what}.<br>Previous version kept as ${j.backup}. Recompile the book to see it on the page.</div>`;
  const btn=document.getElementById('save');btn.textContent='Saved ✓';setTimeout(()=>btn.textContent='Save to .tex',2500);}
 else S.innerHTML=`<div class="banner err">Save failed: ${j.error}</div>`;
 if(j.ok&&EMBED)parent.postMessage({type:'tuner-saved',file:j.file,section:j.section,changed:j.changed,time:j.time},'*');
 if(j.ok){blocks=await (await fetch('/api/blocks')).json();const b=blocks.find(x=>x.section===cur.section&&x.file===cur.file);if(b){cur.id=b.id;cur.start=b.start;cur.end=b.end;cur.tex=b.tex;arrows=b.arrows;cur.arrows=b.arrows;const li=document.querySelector('#list .item.sel');if(li)li.dataset.id=b.id;const keep=S.innerHTML;buildControls();document.getElementById('saved').innerHTML=keep;}}}
load();
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/tuner"):
            data = HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        elif path == "/api/blocks":
            blocks = find_blocks()
            for b in blocks:
                b["arrows"] = parse_arrows(b["tex"])
            self._json(blocks)
        elif path in ("/fonts/Rubik-Regular.ttf", "/fonts/Rubik-Italic.ttf"):
            data = (BOOK / "fonts" / "RubikFont" / path.split("/")[-1]).read_bytes()
            self.send_response(200); self.send_header("Content-Type", "font/ttf"); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode()
        if self.path == "/api/reparse":
            self._json(parse_arrows(body))
        elif self.path == "/api/base":
            ensure_format()
            self._json(base_for(json.loads(body)["tex"]))
        elif self.path == "/api/render":
            ensure_format()
            req = json.loads(body)
            tex = apply_arrows(req["tex"], req["arrows"]) if req.get("arrows") else req["tex"]
            t0 = time.time()
            entry = base_for(tex)
            if "error" in entry:
                self._json({"error": entry["error"], "tex": tex}); return
            png, err = render(tex)
            if png is None:
                self._json({"error": err, "tex": tex})
            else:
                self._json({"png": base64.b64encode(crop_to(png, entry["box"])).decode(), "ms": int((time.time() - t0) * 1000), "tex": tex})
        elif self.path == "/api/save":
            req = json.loads(body)
            tex = apply_arrows(req["tex"], req["arrows"]) if req.get("arrows") else req["tex"]
            fname, start = req["id"].split(":")
            path = BOOK / fname
            text = path.read_text()
            blocks = [b for b in find_blocks() if b["file"] == fname]
            target = next((b for b in blocks if b["start"] == int(start)), None)
            if target is None:
                self._json({"ok": False, "error": "block moved; reload the page"}); return
            new = text[: target["start"]] + tex + text[target["end"]:]
            changed = changed_arrows(target["tex"], tex)
            shutil.copy(path, path.with_suffix(".tex.bak"))
            path.write_text(new)
            self._json({"ok": True, "file": fname, "section": target["section"], "changed": changed,
                        "backup": fname + ".bak", "time": time.strftime("%H:%M:%S")})
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    print(f"Formula tuner: http://localhost:{PORT}   (work dir {WORK})")
    print("building the preamble format…", end=" ", flush=True)
    ensure_format()
    print("ok" if FMT["path"] else "not available, using full compiles")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
