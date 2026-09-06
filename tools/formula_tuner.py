"""Interactive tuner for the book's annotated formulas.

Every formula in the book is a TikZ block: a node holding the math and one
`\\draw` arrow per colored symbol.  Getting the arrow start, end, bend and label
right takes trial and error, so this tool does the loop for you:

    uv run python tools/formula_tuner.py            # then open http://localhost:8765

Left: every formula block in book/*.tex, grouped by chapter.  Middle: the block
compiled on its own (same preamble and fonts as the book, rendered by XeLaTeX
and pdftoppm).  Right: one card per arrow with sliders for the start offset,
the end offset, the bend and the label side, plus the label text.  Moving a
slider re-renders; "Save to .tex" writes the block back into the chapter file
exactly where it came from.  Nothing else in the file is touched.

Requirements: xelatex and pdftoppm on PATH (the book's own toolchain).
"""
import http.server
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.parse
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
    r"to(?:\[bend (?P<bdir>left|right)\s*=\s*(?P<bend>-?\d+)\])?\s*node\[pos=1,\s*(?P<side>left|right|above|below)\]\s*\{(?P<label>.*?)\}\s*"
    r"\+\((?P<ex>-?[\d.]+)\s*,\s*(?P<ey>-?[\d.]+)\)\s*;",
    re.S,
)


def preamble():
    main = (BOOK / "main.tex").read_text()
    return main[: main.index("\\begin{document}")]


PREAMBLE = preamble()


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
            blocks.append({
                "id": f"{f.name}:{m.start()}",
                "file": f.name,
                "start": m.start(),
                "end": m.end(),
                "section": sec,
                "tex": m.group(0),
            })
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
            "side": d["side"], "label": d["label"],
            "ex": float(d["ex"]), "ey": float(d["ey"]),
        })
    return arrows


def arrow_tex(a):
    bend = f"to[bend {a['bdir']}={a['bend']}]" if int(a["bend"]) else "to"
    return (f"\\draw[{a['opts']}] ($({a['anchor']})+({a['sx']:.2f},{a['sy']:.2f})$) {bend} "
            f"node[pos=1, {a['side']}] {{{a['label']}}} +({a['ex']:.2f},{a['ey']:.2f});")


def apply_arrows(tex, arrows):
    """Rebuild the block text with the given arrow parameters (spans from the original)."""
    out = tex
    for a in sorted(arrows, key=lambda x: -x["span"][0]):
        out = out[: a["span"][0]] + arrow_tex(a) + out[a["span"][1]:]
    return out


def changed_arrows(old, new):
    """Labels of arrows whose parameters differ between two block texts, plus 'block text' if anything else differs."""
    def strip(t):
        out = t
        for a in sorted(parse_arrows(t), key=lambda x: -x["span"][0]):
            out = out[: a["span"][0]] + out[a["span"][1]:]
        return re.sub(r"\s+", "", out)
    def vals(a):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items() if k != "span"}
    ao, an = parse_arrows(old), parse_arrows(new)
    changed = [an[i]["label"] for i in range(min(len(ao), len(an))) if vals(ao[i]) != vals(an[i])]
    if len(ao) != len(an) or strip(old) != strip(new):
        changed.append("block text")
    return changed


RENDER_LOCK = threading.Lock()
WORK = Path(tempfile.mkdtemp(prefix="formula_tuner_"))


def render(tex):
    """Compile one block with the book preamble; return PNG bytes or an error string."""
    doc = (PREAMBLE + "\n\\begin{document}\n\\pagestyle{empty}\n\\vspace*{2.2cm}\n" + tex +
           "\n\\vspace*{2.2cm}\n\\end{document}\n")
    with RENDER_LOCK:
        for old in WORK.glob("f.*"):
            old.unlink()
        (WORK / "f.tex").write_text(doc)
        # run inside book/ so the relative font paths resolve; output goes to WORK
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                            f"-output-directory={WORK}", str(WORK / "f.tex")],
                           cwd=BOOK, capture_output=True, text=True, timeout=180)
        pdf = WORK / "f.pdf"
        if not pdf.exists():
            log = (WORK / "f.log").read_text(errors="replace") if (WORK / "f.log").exists() else r.stdout
            err = "\n".join(l for l in log.splitlines() if l.startswith("!") or l.startswith("l."))
            return None, err or log[-2000:]
        subprocess.run(["pdftoppm", "-png", "-r", "170", "-f", "1", "-l", "1", "-singlefile", str(pdf), str(WORK / "f")],
                       check=True, capture_output=True)
        return (WORK / "f.png").read_bytes(), None


def crop_png(png):
    """Trim white margins so the formula fills the viewer (PIL is in the book's venv)."""
    try:
        from PIL import Image, ImageChops
        import io
        im = Image.open(io.BytesIO(png)).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bbox = ImageChops.difference(im, bg).getbbox()
        if bbox:
            pad = 40
            bbox = (max(bbox[0] - pad, 0), max(bbox[1] - pad, 0), min(bbox[2] + pad, im.width), min(bbox[3] + pad, im.height))
            im = im.crop(bbox)
        out = io.BytesIO(); im.save(out, format="PNG"); return out.getvalue()
    except Exception:
        return png


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Formula tuner</title>
<style>
 body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;display:grid;grid-template-columns:260px 1fr 380px;height:100vh;color:#222}
 body.embed{grid-template-columns:1fr 380px}
 body.embed #list{display:none}
 #list{overflow:auto;border-right:1px solid #ddd;padding:8px;font-size:13px}
 #list h4{margin:10px 0 4px;color:#666;font-weight:600;font-size:12px;text-transform:uppercase}
 #list div.item{padding:4px 6px;border-radius:4px;cursor:pointer}
 #list div.item:hover{background:#eef}
 #list div.item.sel{background:#0AA7D4;color:#fff}
 #view{display:flex;flex-direction:column;align-items:center;padding:12px;overflow:auto;background:#fafafa}
 #view img{max-width:100%;border:1px solid #ddd;background:#fff}
 #status{font-size:12px;color:#777;margin:6px}
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
<div id="view"><div id="status">Pick a formula on the left.</div><img id="img" alt=""><pre class="err" id="err"></pre></div>
<div id="ctrl"></div>
<script>
const COLORS={nmlred:'#DD4040',nmlcyan:'#00C8E5',nmlpurple:'#3B0280',nmlgreen:'#4EB046',nmlyellow:'#E1BC29'};
let blocks=[],cur=null,arrows=[],timer=null;
const Q=new URLSearchParams(location.search),EMBED=Q.get('embed')==='1';if(EMBED)document.body.classList.add('embed');
async function load(){blocks=await (await fetch('/api/blocks')).json();const L=document.getElementById('list');let f='';
 for(const b of blocks){if(b.file!==f){f=b.file;const h=document.createElement('h4');h.textContent=f;L.appendChild(h);}
  const d=document.createElement('div');d.className='item';d.textContent=b.section;d.onclick=()=>select(b,d);d.dataset.id=b.id;L.appendChild(d);}
 const want=Q.get('block');if(want){const b=blocks.find(x=>x.id===want);if(b)select(b,document.querySelector(`[data-id="${b.id}"]`));}}
function select(b,el){document.querySelectorAll('#list .item').forEach(x=>x.classList.remove('sel'));if(el)el.classList.add('sel');
 cur=JSON.parse(JSON.stringify(b));arrows=cur.arrows;buildControls();render();}
function color(opts){for(const k in COLORS){if(opts.includes(k))return COLORS[k];}return '#888';}
function slider(card,a,key,min,max,step,name){const r=document.createElement('div');r.className='row';
 r.innerHTML=`<span>${name}</span><input type="range" min="${min}" max="${max}" step="${step}" value="${a[key]}"><input type="text" value="${a[key]}">`;
 const [lab,rng,txt]=r.children;rng.oninput=()=>{a[key]=parseFloat(rng.value);txt.value=rng.value;schedule();};
 txt.onchange=()=>{a[key]=parseFloat(txt.value);rng.value=txt.value;schedule();};card.appendChild(r);}
function buildControls(){const C=document.getElementById('ctrl');C.innerHTML='';
 const top=document.createElement('div');top.innerHTML=`<div style="margin-bottom:8px"><b>${cur.section}</b> <span style="color:#777">${cur.file}</span></div>
 <button class="primary" id="save">Save to .tex</button><button id="rerender">Re-render</button><button id="reset">Reset</button>${EMBED?'<button id="close" style="float:right">Close</button>':''}<div id="saved" style="color:#2a7;font-size:12px;margin-top:4px"></div>`;C.appendChild(top);
 if(EMBED)top.querySelector('#close').onclick=()=>parent.postMessage('tuner-close','*');
 top.querySelector('#save').onclick=save;top.querySelector('#rerender').onclick=render;
 top.querySelector('#reset').onclick=()=>{const b=blocks.find(x=>x.id===cur.id);select(b,document.querySelector(`[data-id="${b.id}"]`));};
 arrows.forEach((a,i)=>{const card=document.createElement('div');card.className='card';
  card.innerHTML=`<div class="lab"><span class="swatch" style="background:${color(a.opts)}"></span>arrow ${i+1}: <input type="text" value="${a.label.replace(/"/g,'&quot;')}" style="width:190px"></div>`;
  card.querySelector('input').onchange=e=>{a.label=e.target.value;schedule();};
  const anc=document.createElement('div');anc.className='row';anc.innerHTML=`<span>anchor</span><select><option>a.north</option><option>a.south</option><option>a.east</option><option>a.west</option><option>b.south</option><option>b.north</option></select><span></span>`;
  const sel=anc.querySelector('select');sel.value=a.anchor;sel.onchange=()=>{a.anchor=sel.value;schedule();};card.appendChild(anc);
  slider(card,a,'sx',-6,6,0.05,'start x');slider(card,a,'sy',-1.5,1.5,0.05,'start y');
  slider(card,a,'ex',-3,3,0.05,'end dx');slider(card,a,'ey',-1.5,1.5,0.05,'end dy');
  slider(card,a,'bend',0,60,1,'bend °');
  const side=document.createElement('div');side.className='row';side.innerHTML=`<span>bend / label</span><select><option value="left">bend left</option><option value="right">bend right</option></select><select><option>left</option><option>right</option><option>above</option><option>below</option></select>`;
  const [s1,s2]=side.querySelectorAll('select');s1.value=a.bdir;s2.value=a.side;s1.onchange=()=>{a.bdir=s1.value;schedule();};s2.onchange=()=>{a.side=s2.value;schedule();};card.appendChild(side);
  C.appendChild(card);});
 const raw=document.createElement('div');raw.className='card';raw.innerHTML='<div class="lab">Raw block (edits here win over the sliders)</div><textarea id="raw"></textarea><button id="applyraw">Apply raw</button>';
 raw.querySelector('#raw').value=cur.tex;raw.querySelector('#applyraw').onclick=()=>{cur.tex=document.getElementById('raw').value;fetch('/api/reparse',{method:'POST',body:cur.tex}).then(r=>r.json()).then(j=>{cur.arrows=j;arrows=j;buildControls();render();});};
 C.appendChild(raw);}
function schedule(){clearTimeout(timer);timer=setTimeout(render,350);}
async function render(){if(!cur)return;document.getElementById('status').textContent='rendering…';document.getElementById('err').textContent='';
 const r=await fetch('/api/render',{method:'POST',body:JSON.stringify({tex:cur.tex,arrows:arrows})});const j=await r.json();
 if(j.png){document.getElementById('img').src='data:image/png;base64,'+j.png;document.getElementById('status').textContent=`rendered in ${j.ms} ms`;document.getElementById('raw').value=j.tex;}
 else{document.getElementById('status').textContent='compile error';document.getElementById('err').textContent=j.error;}}
async function save(){const r=await fetch('/api/save',{method:'POST',body:JSON.stringify({id:cur.id,tex:cur.tex,arrows:arrows})});const j=await r.json();
 const S=document.getElementById('saved');
 if(j.ok){const what=j.changed.length?`${j.changed.length} arrow${j.changed.length>1?'s':''} changed: ${j.changed.map(l=>'“'+l+'”').join(', ')}`:'nothing changed since the last save';
  S.innerHTML=`<div class="banner">&#10003; Saved to <b>${j.file}</b> (${j.section}) at ${j.time}.<br>${what}.<br>Previous version kept as ${j.backup}. Recompile the book to see it on the page.</div>`;
  const btn=document.getElementById('save');btn.textContent='Saved ✓';setTimeout(()=>btn.textContent='Save to .tex',2500);}
 else S.innerHTML=`<div class="banner err">Save failed: ${j.error}</div>`;
 if(j.ok&&EMBED)parent.postMessage({type:'tuner-saved',file:j.file,section:j.section,changed:j.changed,time:j.time},'*');
 if(j.ok){blocks=await (await fetch('/api/blocks')).json();const b=blocks.find(x=>x.section===cur.section&&x.file===cur.file);if(b){cur.id=b.id;cur.start=b.start;cur.end=b.end;cur.tex=b.tex;arrows=b.arrows;cur.arrows=b.arrows;document.querySelector('#list .item.sel').dataset.id=b.id;buildControls();}}}
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
        if self.path.split("?")[0] in ("/", "/tuner"):
            data = HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        elif self.path == "/api/blocks":
            blocks = find_blocks()
            for b in blocks:
                b["arrows"] = parse_arrows(b["tex"])
            self._json(blocks)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode()
        if self.path == "/api/reparse":
            self._json(parse_arrows(body))
        elif self.path == "/api/render":
            import base64, time
            req = json.loads(body)
            tex = apply_arrows(req["tex"], req["arrows"]) if req.get("arrows") else req["tex"]
            t0 = time.time()
            png, err = render(tex)
            if png is None:
                self._json({"error": err, "tex": tex})
            else:
                self._json({"png": base64.b64encode(crop_png(png)).decode(), "ms": int((time.time() - t0) * 1000), "tex": tex})
        elif self.path == "/api/save":
            req = json.loads(body)
            tex = apply_arrows(req["tex"], req["arrows"]) if req.get("arrows") else req["tex"]
            fname, start = req["id"].split(":")
            path = BOOK / fname
            text = path.read_text()
            # locate the block again by its original text to be safe against shifts
            blocks = [b for b in find_blocks() if b["file"] == fname]
            target = next((b for b in blocks if b["start"] == int(start)), None)
            if target is None:
                self._json({"ok": False, "error": "block moved; reload the page"}); return
            new = text[: target["start"]] + tex + text[target["end"]:]
            changed = changed_arrows(target["tex"], tex)
            shutil.copy(path, path.with_suffix(".tex.bak"))
            path.write_text(new)
            import datetime
            self._json({"ok": True, "file": fname, "section": target["section"], "changed": changed,
                        "backup": fname + ".bak", "time": datetime.datetime.now().strftime("%H:%M:%S")})
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    print(f"Formula tuner: http://localhost:{PORT}   (work dir {WORK})")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
