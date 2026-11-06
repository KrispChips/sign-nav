"""Is this crop really a sign? Runs once per track on its best rectified crop, between the
tracker and the VLM, to save Gemma calls on lights, outlets, paintings and the like.

  keep, reason, ocr_text = sign_filter(crop)

Stage 1 (distractor phrases in the detector prompt) lives in detector.py.
Stage 2: PP-OCR text detection + recognition (RapidOCR on ONNX Runtime, no PaddlePaddle).
         Keep if confident lines hold at least `min_chars` letters/digits ("B2" passes,
         an outlet read as "O" doesn't). No text boxes = fast reject, the recognizer never runs.
Stage 3: CLIP ViT-B/32, only for crops that failed OCR, to rescue symbol-only signs
         (restroom pictograms, exit running man, elevator/stairs icons).
         Keep if sum(P(positive prompts)) - sum(P(negative prompts)) > `clip_margin`.

Recall matters more than precision: a missed exit sign is worse than a wasted Gemma call.
When in doubt, loosen the thresholds."""
import re
import time

import cv2

POSITIVE_PROMPTS = (
    "a restroom sign with a pictogram",
    "an exit sign",
    "a wayfinding sign with an arrow",
    "a room number sign next to a door",
    "an elevator sign",
    "a stairs sign",
)
NEGATIVE_PROMPTS = (
    "a light panel", "a ceiling light", "an electrical outlet", "a light switch",
    "a painting", "a picture frame", "a window", "an air vent",
    "a blank white wall", "a door", "a bulletin board",
)


class OcrCheck:
    def __init__(self, min_conf=0.6, min_chars=2, side_len=320):
        from rapidocr import RapidOCR
        # The resize limit is the setting that matters: the default scales the SHORT side up
        # to 736 px, ~10x slower on small crops. The angle classifier is off since crops are
        # already rectified.
        self.engine = RapidOCR(params={"Det.limit_type": "max", "Det.limit_side_len": side_len,
                                       "Global.use_cls": False})
        self.min_conf, self.min_chars = min_conf, min_chars

    def __call__(self, crop):
        r = self.engine(crop, use_cls=False)
        lines = list(zip(r.txts or (), r.scores or ()))
        n = sum(len(re.findall(r"[0-9A-Za-z]", t)) for t, c in lines if c >= self.min_conf)
        text = " | ".join(f"{t} ({c:.2f})" for t, c in lines)
        return n >= self.min_chars, n, text


class ClipCheck:
    def __init__(self, model_id="openai/clip-vit-base-patch32", margin=0.0, device=None,
                 positives=POSITIVE_PROMPTS, negatives=NEGATIVE_PROMPTS):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch, self.margin, self.n_pos = torch, margin, len(positives)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        with torch.no_grad():   # text embeddings once, at startup
            t = self.proc(text=list(positives) + list(negatives), return_tensors="pt",
                          padding=True).to(self.device)
            self.text = self._unit(self.model.get_text_features(**t))

    def _unit(self, out):
        # transformers 4.x returns the projected embedding, 5.x a model output holding it.
        e = out if self.torch.is_tensor(out) else out.pooler_output
        return e / e.norm(dim=-1, keepdim=True)

    def __call__(self, crop):
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with self.torch.no_grad():
            px = self.proc(images=img, return_tensors="pt").to(self.device)
            emb = self._unit(self.model.get_image_features(**px))
            p = (self.model.logit_scale.exp() * emb @ self.text.T).softmax(-1)[0]
        score = float(p[:self.n_pos].sum() - p[self.n_pos:].sum())
        return score > self.margin, score


class SignFilter:
    """crop -> (keep, reason, ocr_text). `stats` counts decisions for the run summary."""

    def __init__(self, ocr=None, clip=None):
        self.ocr, self.clip = ocr, clip
        self.stats = dict(checked=0, ocr_kept=0, ocr_rejected=0, clip_rescued=0, rejected=0)
        self.last_ms = 0.0

    def __call__(self, crop):
        t0 = time.perf_counter()
        self.stats["checked"] += 1
        keep, n, text = self.ocr(crop)
        reason = f"ocr:{n} chars"
        if keep:
            self.stats["ocr_kept"] += 1
        else:
            self.stats["ocr_rejected"] += 1
            if self.clip is not None:
                keep, score = self.clip(crop)
                reason += f", clip:{score:+.2f}"
                if keep:
                    self.stats["clip_rescued"] += 1
            if not keep:
                self.stats["rejected"] += 1
        self.last_ms = 1000 * (time.perf_counter() - t0)
        return keep, reason, text
