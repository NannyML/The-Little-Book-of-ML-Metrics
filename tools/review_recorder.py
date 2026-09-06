"""Review recorder: read the book page by page, leave spoken notes, and tune the formulas in place.

    uv run python tools/review_recorder.py            # open http://localhost:8766
    uv run --with mlx-whisper python tools/review_recorder.py --transcribe   # re-transcribe saved audio locally (optional)

Left: the rendered page (arrow keys to move, dropdown to jump to a metric).
Right: press the button or the space bar, talk, press again. The browser's speech
recognition gives a live transcript; the raw audio is kept as well. Every note is
tagged with the page, chapter and metric it was recorded on and written to

    review/notes.jsonl      one JSON line per note (source of truth)
    review/audio/<id>.webm  the recording
    review/NOTES.md         all notes grouped by chapter and metric, regenerated on every save

Hand NOTES.md to a Claude session ("apply review/NOTES.md") to have the notes worked through metric by metric.

Pages with an annotated formula show a "Tune arrows" button: it opens the formula tuner
(tools/formula_tuner.py) for that formula; "Save to .tex" writes the block back and
"Recompile book" rebuilds main.pdf so the page image catches up.
"""
import base64
import datetime as dt
import http.server
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formula_tuner  # noqa: E402  (shares its render/save endpoints and the embedded tuner page)

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"
PDF = BOOK / "main.pdf"
TOC = BOOK / "main.toc"
REVIEW = ROOT / "review"
AUDIO = REVIEW / "audio"
NOTES = REVIEW / "notes.jsonl"
NOTES_MD = REVIEW / "NOTES.md"
PORT = 8766
DPI = 100
CACHE = Path(tempfile.mkdtemp(prefix="review_pages_"))
COMPILE = {"running": False, "log": "", "ok": None}


def clean_title(t):
    t = re.sub(r"\\tocmark \{\w+\}|\\numberline \{[\d.]+\}|\\texorpdfstring.*?\{\}", "", t)
    t = re.sub(r"\{\\o\s*\}|\\o\s", "ø", t)
    return t.replace("\\&", "&").replace("--", "–").strip()


# ---------------------------------------------------------------------------
# Book structure: physical page -> chapter, section
# ---------------------------------------------------------------------------
def n_pages():
    out = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", out).group(1))


def toc():
    """[(kind, title, page)] in book order, from main.toc (page numbers there are physical pages)."""
    entries = []
    for m in re.finditer(r"\\contentsline \{(chapter|section)\}\{(.*?)\}\{(\d+)\}", TOC.read_text()):
        kind, title, page = m.groups()
        entries.append((kind, clean_title(title), int(page)))
    return entries


def page_index():
    """page -> {'chapter', 'section', 'section_page'} for every physical page."""
    idx, chapter, section, spage = {}, "Front matter", "", 0
    entries = toc()
    total = n_pages()
    by_page = {}
    for kind, title, page in entries:
        by_page.setdefault(page, []).append((kind, title))
    for p in range(1, total + 1):
        for kind, title in by_page.get(p, []):
            if kind == "chapter":
                chapter, section, spage = title, "", p
            else:
                section, spage = title, p
        label = section or ("opener" if chapter == "Introduction" else "opener / decision map" if chapter not in ("Front matter", "Sources") else "")
        idx[p] = {"chapter": chapter, "section": label, "section_page": spage}
    return idx


def blocks_for_page(p):
    """Formula blocks (tuner ids) that belong to the metric on page p."""
    meta = page_index().get(p)
    if not meta or not meta["section"]:
        return []
    return [{"id": b["id"], "section": clean_title(b["section"]), "file": b["file"]}
            for b in formula_tuner.find_blocks() if clean_title(b["section"]) == meta["section"]]


def compile_book():
    if COMPILE["running"]:
        return
    COMPILE.update(running=True, log="", ok=None)

    def run():
        r = subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "main.tex"], cwd=BOOK, capture_output=True, text=True)
        log = (BOOK / "main.log").read_text(errors="replace") if (BOOK / "main.log").exists() else r.stdout
        errs = [l for l in log.splitlines() if l.startswith("! ")]
        for f in CACHE.glob("p-*.png"):
            f.unlink()
        COMPILE.update(running=False, ok=(r.returncode == 0 and not errs), log="\n".join(errs[:10]) or "ok")

    threading.Thread(target=run, daemon=True).start()


def render_page(p):
    out = CACHE / f"p-{p:03d}.png"
    if not out.exists():
        subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-f", str(p), "-l", str(p), "-singlefile", str(PDF), str(CACHE / f"p-{p:03d}")], check=True)
    return out.read_bytes()


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
def load_notes():
    if not NOTES.exists():
        return []
    return [json.loads(l) for l in NOTES.read_text().splitlines() if l.strip()]


def save_notes(notes):
    REVIEW.mkdir(exist_ok=True)
    NOTES.write_text("".join(json.dumps(n, ensure_ascii=False) + "\n" for n in notes))
    write_markdown(notes)


def write_markdown(notes):
    idx = page_index()
    groups = {}
    for n in notes:
        meta = idx.get(n["page"], {"chapter": "?", "section": "?", "section_page": 0})
        groups.setdefault(meta["chapter"], {}).setdefault((meta["section_page"], meta["section"]), []).append(n)
    lines = ["# Review notes", "", f"{len(notes)} notes, regenerated {dt.datetime.now():%Y-%m-%d %H:%M}. Source of truth: `review/notes.jsonl`.", ""]
    for chapter in sorted(groups, key=lambda c: min(k[0] for k in groups[c])):
        lines += [f"## {chapter}", ""]
        for (spage, section), ns in sorted(groups[chapter].items()):
            lines.append(f"### {section or chapter} (p. {spage})" if section else f"### {chapter} (p. {spage})")
            lines.append("")
            for n in sorted(ns, key=lambda n: (n["page"], n["ts"])):
                text = (n.get("text") or "").strip() or "(no transcript yet)"
                src = " typed" if n.get("typed") else f" audio: `review/audio/{n['id']}.webm`"
                lines.append(f"- **p. {n['page']}** ({n['ts'][:16].replace('T', ' ')},{src})  \n  {text}")
            lines.append("")
    NOTES_MD.write_text("\n".join(lines))


def transcribe_missing(all_notes=False):
    import mlx_whisper  # uv run --with mlx-whisper ...
    notes = load_notes()
    todo = [n for n in notes if not n.get("typed") and (all_notes or not (n.get("text") or "").strip())]
    print(f"transcribing {len(todo)} notes with whisper-large-v3-turbo")
    for n in todo:
        path = AUDIO / f"{n['id']}.webm"
        if not path.exists():
            continue
        wav = CACHE / f"{n['id']}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ar", "16000", "-ac", "1", str(wav)], check=True)
        res = mlx_whisper.transcribe(str(wav), path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="en")
        n["text"] = res["text"].strip()
        n["whisper"] = True
        print(f"  p.{n['page']}: {n['text'][:80]}")
    save_notes(notes)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Book review recorder</title>
<style>
  body{margin:0;font:14px/1.4 -apple-system,Helvetica,Arial,sans-serif;color:#222;background:#f4f4f4;display:grid;grid-template-columns:minmax(0,1fr) 400px;height:100vh}
  #left{display:flex;flex-direction:column;align-items:center;overflow:auto;padding:12px}
  #page{max-height:calc(100vh - 70px);box-shadow:0 2px 12px rgba(0,0,0,.18);background:#fff}
  #nav{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap;justify-content:center}
  #right{background:#fff;border-left:1px solid #ddd;padding:16px;display:flex;flex-direction:column;gap:12px;overflow:auto}
  button{font:inherit;padding:6px 12px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
  button:hover{background:#f0f0f0}
  #rec{font-size:18px;padding:16px;border-radius:10px;background:#00c8e5;border-color:#00c8e5;color:#fff;font-weight:600}
  #rec.on{background:#dd4040;border-color:#dd4040;animation:pulse 1.2s infinite}
  @keyframes pulse{50%{opacity:.7}}
  #where{color:#555}
  #where b{color:#222}
  #live{min-height:60px;border:1px dashed #ccc;border-radius:6px;padding:8px;color:#333;white-space:pre-wrap}
  #live.interim{color:#888}
  .note{border:1px solid #e3e3e3;border-radius:6px;padding:8px;margin-bottom:6px;background:#fafafa;position:relative}
  .note small{color:#888}
  .note .del{position:absolute;right:6px;top:6px;border:none;background:none;color:#aaa;font-size:16px;padding:0 4px}
  .note .del:hover{color:#dd4040}
  .note textarea{width:100%;box-sizing:border-box;border:none;background:transparent;font:inherit;resize:vertical;min-height:40px}
  #typed{width:100%;box-sizing:border-box;font:inherit;padding:6px;border:1px solid #ccc;border-radius:6px;min-height:50px}
  h3{margin:8px 0 4px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#777}
  select{font:inherit;padding:5px;max-width:280px}
  #status{color:#888;font-size:12px}
  a{color:#0077aa}
  #formulas button{margin:4px 6px 0 0;border-color:#3b0280;color:#3b0280}
  #overlay{position:fixed;inset:0;background:#fff;z-index:10}
  #overlay iframe{width:100%;height:100%;border:none}
</style>
<div id="left">
  <div id="nav">
    <button onclick="go(page-1)">&larr;</button>
    <span id="pno"></span>
    <button onclick="go(page+1)">&rarr;</button>
    <select id="jump" onchange="go(+this.value)"></select>
  </div>
  <img id="page">
</div>
<div id="right">
  <div id="where"></div>
  <button id="rec" onclick="toggle()">&#9679; Record (space)</button>
  <div id="status">Chrome or Safari; allow the microphone when asked.</div>
  <div id="live" class="interim">Live transcript appears here while you talk.</div>
  <div id="formulas"></div>
  <h3>Or type</h3>
  <textarea id="typed" placeholder="Type a note and press Cmd+Enter"></textarea>
  <h3>Notes on this page</h3>
  <div id="notes"></div>
  <div id="status2" style="color:#888;font-size:12px"><span id="count"></span> notes in total &middot; <a href="/notes.md" target="_blank">NOTES.md</a></div>
  <div style="margin-top:auto;padding-top:12px;border-top:1px solid #eee;font-size:12px;color:#888"><button id="compile" onclick="compileBook()">Recompile book</button> <span id="cstatus">after saving a formula, so the page image catches up (about a minute)</span></div>
</div>
<div id="overlay" hidden><iframe id="tuner"></iframe></div>
<script>
let page = +localStorage.getItem('review_page') || 1, total = 1, index = {}, notes = [];
let recording = false, rec = null, chunks = [], recog = null, finalText = '', interimText = '', stream = null;

async function init(){
  const r = await fetch('/api/index'); const d = await r.json();
  total = d.total; index = d.index;
  const sel = document.getElementById('jump');
  for (const e of d.toc){ const o = document.createElement('option'); o.value = e.page; o.textContent = (e.kind==='chapter'? '— ' : '      ') + e.title + '  (p. ' + e.page + ')'; sel.appendChild(o); }
  await loadNotes(); go(page);
}
async function loadNotes(){ notes = await (await fetch('/api/notes')).json(); document.getElementById('count').textContent = notes.length; }
function go(p){
  if (p < 1 || p > total || recording) return;
  page = p; localStorage.setItem('review_page', p);
  document.getElementById('page').src = '/page/' + p + '.png';
  document.getElementById('pno').textContent = 'page ' + p + ' / ' + total;
  const m = index[p]; document.getElementById('where').innerHTML = '<b>' + m.chapter + '</b>' + (m.section ? ' &rsaquo; <b>' + m.section + '</b>' : '') + ' &middot; page ' + p;
  const sel = document.getElementById('jump'); let best = null;
  for (const o of sel.options) if (+o.value <= p) best = o; if (best) sel.value = best.value;
  renderNotes(); renderFormulas();
}
function renderNotes(){
  const box = document.getElementById('notes'); box.innerHTML = '';
  for (const n of notes.filter(n => n.page === page)){
    const div = document.createElement('div'); div.className = 'note';
    div.innerHTML = '<small>' + n.ts.slice(0,16).replace('T',' ') + (n.typed ? ' · typed' : ' · audio') + '</small><button class="del" title="delete">&times;</button><textarea></textarea>';
    div.querySelector('textarea').value = n.text || '(no transcript; run --transcribe)';
    div.querySelector('textarea').onchange = e => fetch('/api/edit', {method:'POST', body: JSON.stringify({id:n.id, text:e.target.value})}).then(loadNotes);
    div.querySelector('.del').onclick = () => { if (confirm('Delete this note?')) fetch('/api/delete', {method:'POST', body: JSON.stringify({id:n.id})}).then(loadNotes).then(renderNotes); };
    box.appendChild(div);
  }
}
async function renderFormulas(){
  const box = document.getElementById('formulas'); box.innerHTML = '';
  const bl = await (await fetch('/api/blocks_for?page=' + page)).json();
  if (!bl.length) return;
  box.innerHTML = '<h3>Formula on this page</h3>';
  bl.forEach((b, i) => { const btn = document.createElement('button'); btn.textContent = 'Tune arrows' + (bl.length > 1 ? ' (' + (i+1) + ')' : ''); btn.onclick = () => openTuner(b.id); box.appendChild(btn); });
}
function openTuner(id){ document.getElementById('tuner').src = '/tuner?embed=1&block=' + encodeURIComponent(id); document.getElementById('overlay').hidden = false; }
window.addEventListener('message', e => {
  if (e.data === 'tuner-close'){ document.getElementById('overlay').hidden = true; document.getElementById('tuner').src = 'about:blank'; }
  if (e.data === 'tuner-saved') document.getElementById('cstatus').textContent = 'formula saved to .tex; recompile to see it on the page';
});
async function compileBook(){
  await fetch('/api/compile', {method:'POST', body:'{}'}); document.getElementById('cstatus').textContent = 'compiling…'; document.getElementById('compile').disabled = true;
  const poll = setInterval(async () => { const st = await (await fetch('/api/compile')).json();
    if (!st.running){ clearInterval(poll); document.getElementById('compile').disabled = false; document.getElementById('cstatus').textContent = st.ok ? 'compiled; page images refreshed' : ('compile failed: ' + st.log); document.getElementById('page').src = '/page/' + page + '.png?t=' + Date.now(); } }, 3000);
}
async function toggle(){ recording ? await stop() : await start(); }
async function start(){
  try { stream = await navigator.mediaDevices.getUserMedia({audio:true}); } catch(e){ status('Microphone blocked: ' + e.message); return; }
  chunks = []; finalText = ''; interimText = '';
  rec = new MediaRecorder(stream); rec.ondataavailable = e => chunks.push(e.data); rec.start();
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SR){
    recog = new SR(); recog.continuous = true; recog.interimResults = true; recog.lang = 'en-US';
    recog.onresult = ev => { interimText = ''; for (let i = ev.resultIndex; i < ev.results.length; i++){ const t = ev.results[i][0].transcript; if (ev.results[i].isFinal) finalText += t + ' '; else interimText += t; } live(); };
    recog.onend = () => { if (recording) try { recog.start(); } catch(e){} };
    recog.onerror = ev => status('speech recognition: ' + ev.error + ' (audio is still being recorded)');
    try { recog.start(); } catch(e){}
  } else status('No speech recognition in this browser; audio is recorded, transcribe later with --transcribe.');
  recording = true; document.getElementById('rec').classList.add('on'); document.getElementById('rec').innerHTML = '&#9632; Stop (space)'; live();
}
async function stop(){
  recording = false;
  if (recog){ recog.onend = null; try { recog.stop(); } catch(e){} }
  await new Promise(res => { rec.onstop = res; rec.stop(); });
  stream.getTracks().forEach(t => t.stop());
  document.getElementById('rec').classList.remove('on'); document.getElementById('rec').innerHTML = '&#9679; Record (space)';
  const blob = new Blob(chunks, {type: rec.mimeType || 'audio/webm'});
  const b64 = await new Promise(res => { const fr = new FileReader(); fr.onload = () => res(fr.result.split(',')[1]); fr.readAsDataURL(blob); });
  const text = (finalText + ' ' + interimText).trim();
  await fetch('/api/note', {method:'POST', body: JSON.stringify({page, text, audio: b64, mime: blob.type})});
  await loadNotes(); renderNotes(); status('saved ' + Math.round(blob.size/1024) + ' KB' + (text ? '' : ' (no live transcript; run --transcribe later)'));
  document.getElementById('live').textContent = 'Live transcript appears here while you talk.'; document.getElementById('live').className = 'interim';
}
function live(){ const el = document.getElementById('live'); el.className = ''; el.textContent = (finalText + interimText) || (recording ? 'listening…' : ''); }
function status(s){ document.getElementById('status').textContent = s; }
async function saveTyped(){
  const ta = document.getElementById('typed'); const text = ta.value.trim(); if (!text) return;
  await fetch('/api/note', {method:'POST', body: JSON.stringify({page, text, typed: true})}); ta.value = ''; await loadNotes(); renderNotes();
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA'){ if (e.key === 'Enter' && e.metaKey && e.target.id === 'typed') saveTyped(); return; }
  if (!document.getElementById('overlay').hidden) return;
  if (e.code === 'Space'){ e.preventDefault(); toggle(); }
  else if (e.key === 'ArrowRight') go(page+1); else if (e.key === 'ArrowLeft') go(page-1);
});
init();
</script>
"""


TUNER_PATHS = ("/tuner", "/api/blocks", "/api/reparse", "/api/render", "/api/save")


class Handler(formula_tuner.Handler):
    def _send(self, data, ctype="application/json", code=200):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(HTML.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/index":
            entries = [{"kind": k, "title": t, "page": p} for k, t, p in toc()]
            self._send({"total": n_pages(), "index": page_index(), "toc": entries})
        elif self.path == "/api/notes":
            self._send(load_notes())
        elif self.path == "/notes.md":
            write_markdown(load_notes())
            self._send(NOTES_MD.read_bytes(), "text/plain; charset=utf-8")
        elif m := re.fullmatch(r"/page/(\d+)\.png(\?.*)?", self.path):
            self._send(render_page(int(m.group(1))), "image/png")
        elif m := re.fullmatch(r"/api/blocks_for\?page=(\d+)", self.path):
            self._send(blocks_for_page(int(m.group(1))))
        elif self.path == "/api/compile":
            self._send(COMPILE)
        elif self.path.split("?")[0] in TUNER_PATHS:
            super().do_GET()
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path in TUNER_PATHS:
            return super().do_POST()
        if self.path == "/api/compile":
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            compile_book()
            return self._send({"ok": True})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode() or "{}")
        notes = load_notes()
        if self.path == "/api/note":
            nid = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
            note = {"id": nid, "ts": dt.datetime.now().isoformat(timespec="seconds"), "page": int(req["page"]), "text": req.get("text", "")}
            if req.get("typed"):
                note["typed"] = True
            elif req.get("audio"):
                AUDIO.mkdir(parents=True, exist_ok=True)
                (AUDIO / f"{nid}.webm").write_bytes(base64.b64decode(req["audio"]))
            notes.append(note)
            save_notes(notes)
            self._send({"ok": True, "id": nid})
        elif self.path == "/api/edit":
            for n in notes:
                if n["id"] == req["id"]:
                    n["text"] = req["text"]
            save_notes(notes)
            self._send({"ok": True})
        elif self.path == "/api/delete":
            notes = [n for n in notes if n["id"] != req["id"]]
            (AUDIO / f"{req['id']}.webm").unlink(missing_ok=True)
            save_notes(notes)
            self._send({"ok": True})
        else:
            self._send(b"not found", "text/plain", 404)


if __name__ == "__main__":
    if "--transcribe" in sys.argv:
        transcribe_missing(all_notes="--all" in sys.argv)
        sys.exit(0)
    if not PDF.exists():
        sys.exit("book/main.pdf not found: compile the book first (cd book && latexmk -xelatex main.tex)")
    print(f"Review recorder: http://localhost:{PORT}   notes -> {NOTES_MD}")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
