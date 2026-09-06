"""Compute the model-based numbers used by the GenAI-chapter figures.

This script needs heavy dependencies (torch, transformers, lpips, pesq,
pystoi, bert-score) that are NOT part of the book's uv environment.  It is run
once in a throwaway virtualenv and writes small JSON/PNG artefacts into this
directory; `notebooks/genai_plots.py` only reads those artefacts, so the book
figures stay reproducible without the models.

    uv venv /tmp/mlenv && source /tmp/mlenv/bin/activate
    uv pip install torch torchvision transformers lpips pesq pystoi bert-score pillow scipy
    python notebooks/data/genai/gen_model_data.py

Models used (all public, downloaded from the Hugging Face hub / torchvision):
  gpt2 (124M)                    -> per-token probabilities for perplexity
  bert-base-uncased, layer 9     -> BERTScore token-similarity matrices
  openai/clip-vit-base-patch32   -> CLIP Score image/caption cosines
  Salesforce/blip-vqa-base       -> a VQAScore-style P(yes) for the same pairs
  lpips 'alex'                   -> LPIPS distances for equal-MSE distortions
  ITU-T P.862 (pesq), pystoi     -> PESQ / STOI on macOS `say` speech
"""
import json
import math
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIG = REPO / "book" / "figures"
OUT = HERE
torch.manual_seed(0)
np.random.seed(0)


def dump(name, obj):
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=1)
    print(f"  wrote {name}.json")


# ---------------------------------------------------------------------------
# 1. Perplexity: GPT-2 per-token surprisal
# ---------------------------------------------------------------------------
def perplexity():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    texts = {
        "natural": "The Eiffel Tower is located in Paris, the capital of France.",
        "one_swap": "The Eiffel Tower is located in Ohio, the capital of France.",
        "shuffled": "Tower the located France of Paris, is capital Eiffel The in.",
    }
    res = {}
    for key, text in texts.items():
        ids = tok(text, return_tensors="pt").input_ids
        # prepend BOS so the first real token also gets a probability
        bos = torch.tensor([[tok.bos_token_id]])
        inp = torch.cat([bos, ids], dim=1)
        with torch.no_grad():
            logits = model(inp).logits[0, :-1]
        logp = torch.log_softmax(logits, dim=-1)
        target = inp[0, 1:]
        tok_logp = logp[torch.arange(len(target)), target].numpy()
        surprisal = (-tok_logp).tolist()
        tokens = [tok.decode([t]) for t in target.tolist()]
        ppl = float(math.exp(np.mean(surprisal)))
        res[key] = {"text": text, "tokens": tokens, "surprisal": surprisal, "ppl": ppl}
        print(f"  {key}: PPL={ppl:.1f}  max surprisal={max(surprisal):.1f} nats")
    dump("perplexity", res)


# ---------------------------------------------------------------------------
# 2. BERTScore: bert-base-uncased layer 9 (the reference implementation's
#    default layer for this model), greedy matching, raw + rescaled scores
# ---------------------------------------------------------------------------
def bertscore():
    from transformers import AutoModel, AutoTokenizer
    import bert_score
    name = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, output_hidden_states=True).eval()
    LAYER = 9

    def embed(sent):
        enc = tok(sent, return_tensors="pt")
        with torch.no_grad():
            hs = model(**enc).hidden_states[LAYER][0]
        hs = hs[1:-1]  # drop [CLS] / [SEP]
        hs = hs / hs.norm(dim=-1, keepdim=True)
        toks = tok.convert_ids_to_tokens(enc.input_ids[0])[1:-1]
        return toks, hs

    pairs = {
        "paraphrase": ("the doctor delayed the surgery", "the physician postponed the operation"),
        "unrelated": ("the doctor delayed the surgery", "stock prices fell sharply on tuesday"),
        "role_swap": ("the nurse called the doctor", "the doctor called the nurse"),
        "negation": ("the nurse called the doctor", "the nurse did not call the doctor"),
    }
    out = {"layer": LAYER, "model": name, "pairs": {}}
    for key, (ref, cand) in pairs.items():
        rt, re_ = embed(ref)
        ct, ce = embed(cand)
        sim = (re_ @ ce.T).numpy()          # rows: reference tokens, cols: candidate tokens
        R = float(sim.max(axis=1).mean())
        P = float(sim.max(axis=0).mean())
        F = 2 * P * R / (P + R)
        # cross-check with the reference implementation (raw + baseline-rescaled)
        Pb, Rb, Fb = bert_score.score([cand], [ref], model_type=name, num_layers=LAYER,
                                      rescale_with_baseline=False, lang="en", verbose=False)
        Pr, Rr, Fr = bert_score.score([cand], [ref], model_type=name, num_layers=LAYER,
                                      rescale_with_baseline=True, lang="en", verbose=False)
        # unigram precision (BLEU-1 without brevity penalty) on words
        rw, cw = ref.split(), cand.split()
        bleu1 = sum(min(cw.count(w), rw.count(w)) for w in set(cw)) / len(cw)
        out["pairs"][key] = {
            "ref": ref, "cand": cand, "ref_tokens": rt, "cand_tokens": ct, "sim": sim.tolist(),
            "P": P, "R": R, "F": F,
            "P_lib": float(Pb), "R_lib": float(Rb), "F_lib": float(Fb),
            "P_rescaled": float(Pr), "R_rescaled": float(Rr), "F_rescaled": float(Fr),
            "bleu1": bleu1,
        }
        print(f"  {key:11s} P={P:.3f} R={R:.3f} F={F:.3f} | lib F={float(Fb):.3f} "
              f"rescaled F={float(Fr):.3f} | BLEU-1={bleu1:.2f}")
    dump("bertscore", out)


# ---------------------------------------------------------------------------
# Shared images for CLIP / VQA / LPIPS
# ---------------------------------------------------------------------------
def load_images():
    import sklearn, matplotlib
    sk = Path(sklearn.__file__).parent / "datasets" / "images"
    mp = Path(matplotlib.__file__).parent / "mpl-data" / "sample_data"
    paths = {
        "hopper": mp / "grace_hopper.jpg",
        "temple": sk / "china.jpg",
        "dahlia": sk / "flower.jpg",
        "gift": mp / "Minduka_Present_Blue_Pack.png",
    }
    ims = {}
    for k, p in paths.items():
        im = Image.open(p)
        if im.mode in ("RGBA", "LA", "P"):  # flatten transparency onto white
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        # centre-crop to square and shrink so the figure can embed thumbnails
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((256, 256), Image.LANCZOS)
        im.save(OUT / f"img_{k}.png")
        ims[k] = im
    return ims


CAPTIONS = [
    ("hopper", "a woman in a navy uniform in front of a flag"),
    ("temple", "a pagoda on a hill above a lake"),
    ("dahlia", "an orange dahlia flower"),
    ("gift", "a blue gift box with a ribbon"),
    # probes for word order and negation (both refer to the hopper image)
    ("hopper_swap", "a flag in a navy uniform in front of a woman"),
    ("hopper_neg", "a woman not wearing a uniform"),
]


# ---------------------------------------------------------------------------
# 3. CLIP Score: cosine(image, caption) with openai/clip-vit-base-patch32
# ---------------------------------------------------------------------------
def clip_score(ims):
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    proc = CLIPProcessor.from_pretrained(name)
    keys = list(ims.keys())
    caps = [c for _, c in CAPTIONS]
    inputs = proc(text=caps, images=[ims[k] for k in keys], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    cos = (ie @ te.T).numpy()  # images x captions
    res = {"model": name, "images": keys, "captions": CAPTIONS, "cos": cos.tolist(), "w": 2.5}
    for i, k in enumerate(keys):
        print("  ", k, " ".join(f"{c:.3f}" for c in cos[i]))
    dump("clip", res)


# ---------------------------------------------------------------------------
# 4. VQAScore-style P(yes) with a small VQA model (ViLT).  The paper's model
#    (CLIP-FlanT5) is far larger; the mechanism is identical: ask
#    "Does this figure show '<prompt>'?" and read the probability of "yes".
# ---------------------------------------------------------------------------
def vqa_score(ims):
    from transformers import BlipForQuestionAnswering, BlipProcessor
    name = "Salesforce/blip-vqa-base"
    proc = BlipProcessor.from_pretrained(name)
    model = BlipForQuestionAnswering.from_pretrained(name).eval()
    tok = proc.tokenizer
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    keys = list(ims.keys())
    template = "Does this figure show '{}'? Please answer yes or no."
    res = {"model": name, "images": keys, "captions": CAPTIONS, "p_yes": [], "template": template}
    for k in keys:
        row = []
        for _, cap in CAPTIONS:
            enc = proc(ims[k], template.format(cap), return_tensors="pt")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
            sc = torch.softmax(g.scores[0][0], -1)   # answer decoder, first token
            row.append(float(sc[yes_id] / (sc[yes_id] + sc[no_id])))
        res["p_yes"].append(row)
        print("  ", k, " ".join(f"{c:.3f}" for c in row))
    dump("vqascore", res)


# ---------------------------------------------------------------------------
# 5. LPIPS vs MSE: distortions tuned to the SAME pixel MSE
# ---------------------------------------------------------------------------
def lpips_equal_mse(ims):
    import lpips
    loss = lpips.LPIPS(net="alex", verbose=False)
    crown = Image.open(FIG / "DIV2K_0803.png").convert("RGB")
    w, h = crown.size
    s_ = min(w, h)
    crown = crown.crop(((w - s_) // 2, (h - s_) // 2, (w - s_) // 2 + s_, (h - s_) // 2 + s_)).resize((256, 256), Image.LANCZOS)
    crown.save(OUT / "lpips_original.png")
    base = np.asarray(crown).astype(np.float64) / 255.0

    def to_t(a):
        return torch.tensor(a * 2 - 1, dtype=torch.float32).permute(2, 0, 1)[None]

    def mse(a):
        return float(np.mean((a - base) ** 2))

    def d_lpips(a):
        with torch.no_grad():
            return float(loss(to_t(base), to_t(a)))

    def shift(px):
        return np.roll(base, px, axis=1)

    def blur(r):
        return np.asarray(crown.filter(ImageFilter.GaussianBlur(r))).astype(np.float64) / 255.0

    def noise(sd, rng=np.random.default_rng(1)):
        return np.clip(base + rng.normal(0, sd, base.shape), 0, 1)

    def bright(b):
        return np.clip(base + b, 0, 1)

    # anchor: a 4-pixel horizontal shift
    target = mse(shift(4))

    def tune(fn, lo, hi, increasing=True, n=40):
        # bisection on the parameter so that mse(fn(param)) ~= target
        for _ in range(n):
            mid = (lo + hi) / 2
            m = mse(fn(mid))
            if (m < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    r_blur = tune(blur, 0.1, 30, True)
    sd_noise = tune(noise, 0.001, 1.0, True)
    b_bright = tune(bright, 0.0, 0.6, True)
    variants = {
        "shift": ("shift 4 px", shift(4)),
        "bright": (f"brighten +{b_bright:.2f}", bright(b_bright)),
        "noise": (f"noise σ={sd_noise:.2f}", noise(sd_noise)),
        "blur": (f"blur σ={r_blur:.1f}", blur(r_blur)),
    }
    res = {"target_mse": target, "variants": {}}
    for k, (label, a) in variants.items():
        Image.fromarray((a * 255).round().astype(np.uint8)).save(OUT / f"lpips_{k}.png")
        res["variants"][k] = {"label": label, "mse": mse(a), "psnr": 10 * math.log10(1 / mse(a)),
                              "lpips": d_lpips(a)}
        print(f"  {k:6s} {label:14s} MSE={mse(a):.5f} PSNR={res['variants'][k]['psnr']:.1f} dB "
              f"LPIPS={d_lpips(a):.3f}")
    dump("lpips", res)


# ---------------------------------------------------------------------------
# 6. PESQ / STOI on synthesized speech
# ---------------------------------------------------------------------------
def read_wav(path):
    with wave.open(str(path)) as w:
        fs = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    return fs, x


def snr_db(clean, deg):
    return 10 * math.log10(np.sum(clean ** 2) / np.sum((clean - deg) ** 2))


def add_noise(clean, snr, rng):
    n = rng.normal(0, 1, len(clean))
    n *= np.sqrt(np.sum(clean ** 2) / (np.sum(n ** 2) * 10 ** (snr / 10)))
    return clean + n


def audio_metrics():
    from pesq import pesq
    from pystoi import stoi
    from scipy.signal import butter, sosfilt, resample_poly
    src = Path(os.environ.get("SPEECH_WAV", HERE / "speech.wav"))
    fs, clean = read_wav(src)
    assert fs == 16000
    rng = np.random.default_rng(3)
    conds = {}
    conds["clean"] = ("clean", clean.copy())
    d = int(0.020 * fs)
    conds["delay"] = ("delayed 20 ms", np.concatenate([np.zeros(d), clean[:-d]]))
    conds["gain"] = ("−6 dB gain", clean * 10 ** (-6 / 20))
    sos = butter(6, [300, 3400], btype="band", fs=fs, output="sos")
    conds["telephone"] = ("300–3400 Hz band", sosfilt(sos, clean))
    for s in (40, 30, 20, 10, 0, -5):
        conds[f"noise{s}"] = (f"white noise {s} dB SNR", add_noise(clean, s, rng))
    res = {}
    for k, (label, deg) in conds.items():
        deg = np.clip(deg, -1, 1)
        p_wb = float(pesq(fs, clean, deg, "wb"))
        # classic narrowband P.862 at 8 kHz (the P.862.1 MOS-LQO mapping in the book)
        p_nb = float(pesq(8000, resample_poly(clean, 1, 2), resample_poly(deg, 1, 2), "nb"))
        st = float(stoi(clean, deg, fs, extended=False))
        s = snr_db(clean, deg) if k != "clean" else float("inf")
        res[k] = {"label": label, "snr_db": None if s == float("inf") else s,
                  "pesq_wb": p_wb, "pesq_nb": p_nb, "stoi": st}
        print(f"  {label:22s} SNR={s:6.1f} dB  PESQ nb={p_nb:.2f} wb={p_wb:.2f}  STOI={st:.3f}")
        # keep a few degraded signals for the STOI envelope figure
        if k in ("noise0", "noise-5", "noise10"):
            np.save(OUT / f"audio_{k}.npy", deg.astype(np.float32))
    np.save(OUT / "audio_clean.npy", clean.astype(np.float32))
    dump("audio", res)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["perplexity", "bertscore", "images", "clip", "vqa", "lpips", "audio"]
    ims = None
    for s in steps:
        print(f"== {s} ==")
        if s == "perplexity":
            perplexity()
        elif s == "bertscore":
            bertscore()
        elif s in ("images", "clip", "vqa", "lpips"):
            ims = ims or load_images()
            if s == "clip":
                clip_score(ims)
            elif s == "vqa":
                vqa_score(ims)
            elif s == "lpips":
                lpips_equal_mse(ims)
        elif s == "audio":
            audio_metrics()
