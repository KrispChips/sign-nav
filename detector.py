"""Sign detector. Grounding DINO (same family as the Sign-Understanding baseline),
loaded through Hugging Face transformers so there's no repo build step.
Later: replace with a small YOLO trained on auto-labels from this detector."""
import cv2
import numpy as np


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, thr=0.5):
    keep = []
    for box, score in sorted(dets, key=lambda d: -d[1]):
        if all(_iou(box, k[0]) < thr for k in keep):
            keep.append((box, score))
    return keep


class SignDetector:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny",
                 prompt="sign. room number plate. door sign. directional sign.",
                 box_thr=0.30, text_thr=0.25, min_area_px=400, max_area_frac=0.5, device=None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.prompt, self.box_thr, self.text_thr = prompt, box_thr, text_thr
        self.min_area, self.max_area_frac = min_area_px, max_area_frac

    def __call__(self, bgr):
        from PIL import Image
        h, w = bgr.shape[:2]
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, text=self.prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        kw = dict(text_threshold=self.text_thr, target_sizes=[(h, w)])
        try:      # newer transformers
            res = self.processor.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=self.box_thr, **kw)[0]
        except TypeError:  # older transformers
            res = self.processor.post_process_grounded_object_detection(
                out, inputs.input_ids, box_threshold=self.box_thr, **kw)[0]
        dets = []
        for box, s in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()):
            x1, y1, x2, y2 = [float(v) for v in box]
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area or area > self.max_area_frac * w * h:
                continue
            dets.append((np.array([x1, y1, x2, y2]), float(s)))
        return nms(dets)
