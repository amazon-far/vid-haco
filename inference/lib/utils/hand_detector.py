"""
Hand detector: the WiLoR YOLO hand detector, exported to ONNX by the user, run with onnxruntime.

This repository does not ship the detector or the library it was trained with. Users download WiLoR's
`detector.pt` and convert it once, in an environment of their own (see inference/README.md):

    yolo export model=detector.pt format=onnx imgsz=512 opset=17 dynamic=True

The exported graph takes a letterboxed RGB image [1, 3, H, W] in [0, 1] and returns one row per anchor,
[1, 4 + n_classes + extra, A]: box centre x, centre y, width, height (input pixels), one score per class
(WiLoR: 0 = left hand, 1 = right hand), then 21 x (x, y, visibility) hand keypoints, unused here. This
module letterboxes the frame, decodes the boxes, removes overlapping boxes per class (greedy NMS) and
maps the result back to frame pixels.

    det = HandDetector('data/base_data/demo_data/wilor_detector.onnx')
    for x1, y1, x2, y2, score, cls in det(rgb_frame, conf=0.3, iou=0.3): ...

Export with dynamic=True. The graph is then fed the smallest stride-aligned letterbox of the frame at the
detector's own size (512), which reproduces the original detector's boxes to well under a pixel. A
fixed-size export is fed the full padded square, and the detector's boxes change noticeably on
non-square frames, so it is accepted but warned about.
"""
import warnings
import ast
import os

import cv2
import numpy as np

RIGHT, LEFT = 1, 0
PAD_VALUE = 114
MAX_DETECTIONS = 300


def _session(path, providers):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    if providers is None:
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            # onnxruntime-gpu >= 1.21 can load CUDA / cuDNN from the nvidia-* wheels PyTorch installs
            if hasattr(ort, 'preload_dlls'):
                try:
                    ort.preload_dlls()
                except Exception:
                    pass
            providers = ['CUDAExecutionProvider'] + providers
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def _split_padding(total):
    """Split `total` padding pixels between the two sides, the larger half after the image."""
    half = total / 2.0
    before = int(round(half - 0.1))
    return before, int(round(half + 0.1))


def _nms(boxes, scores, iou):
    """Greedy non-maximum suppression. boxes [K,4] xyxy, scores [K] -> kept indices, best first."""
    order = np.argsort(-scores, kind='stable')
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        w = np.maximum(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0)
        h = np.maximum(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0)
        inter = w * h
        overlap = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[overlap <= iou]
    return np.asarray(keep, dtype=np.int64)


class HandDetector:
    def __init__(self, path, size=None, providers=None):
        """size: letterbox target (long side). Default: the export's `imgsz` (512 for WiLoR's detector,
        the size it was trained and is normally run at)."""
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found. Export WiLoR's detector to ONNX first (inference/README.md): "
                f"yolo export model=detector.pt format=onnx imgsz=512 opset=17 dynamic=True")
        self.session = _session(path, providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.dynamic = not (isinstance(h, int) and isinstance(w, int))
        if not self.dynamic:
            warnings.warn(f"{path} has a fixed input size ({h}x{w}); its boxes differ from the detector's "
                          f"usual output on non-square frames. Re-export with dynamic=True.")
        meta = self.session.get_modelmeta().custom_metadata_map
        exported = max(ast.literal_eval(meta['imgsz'])) if 'imgsz' in meta else 512
        self.size = int(size or exported) if self.dynamic else int(max(h, w))
        self.stride = int(meta.get('stride', 32))
        names = ast.literal_eval(meta['names']) if 'names' in meta else {LEFT: 'left', RIGHT: 'right'}
        self.names = {int(k): v for k, v in names.items()}
        self.num_classes = len(self.names)

    @property
    def providers(self):
        return self.session.get_providers()

    def _letterbox(self, rgb):
        h, w = rgb.shape[:2]
        gain = min(self.size / h, self.size / w)
        new_w, new_h = int(round(w * gain)), int(round(h * gain))
        pad_w, pad_h = self.size - new_w, self.size - new_h
        if self.dynamic:                       # smallest stride-aligned canvas around the resized frame
            pad_w, pad_h = pad_w % self.stride, pad_h % self.stride
        left, right = _split_padding(pad_w)
        top, bottom = _split_padding(pad_h)
        img = rgb if (new_w, new_h) == (w, h) else cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return x, gain, left, top

    def __call__(self, rgb, conf=0.3, iou=0.3):
        """rgb: HxWx3 uint8 RGB frame -> list of (x1, y1, x2, y2, score, cls), best first per class."""
        x, gain, left, top = self._letterbox(rgb)
        out = self.session.run(None, {self.input_name: x})[0][0]          # [4 + nc + extra, A]
        cls_scores = out[4:4 + self.num_classes]                            # [nc, A]
        cls = cls_scores.argmax(0)
        score = cls_scores.max(0)
        sel = score > conf
        if not sel.any():
            return []
        cx, cy, bw, bh = out[0, sel], out[1, sel], out[2, sel], out[3, sel]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        score, cls = score[sel], cls[sel]

        keep = []
        for c in np.unique(cls):                                            # NMS within each class
            idx = np.nonzero(cls == c)[0]
            keep.extend(idx[_nms(boxes[idx], score[idx], iou)])
        keep = np.asarray(keep, dtype=np.int64)
        keep = keep[np.argsort(-score[keep], kind='stable')][:MAX_DETECTIONS]

        h, w = rgb.shape[:2]
        b = (boxes[keep] - np.array([left, top, left, top], np.float32)) / gain
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
        return [(float(b[i, 0]), float(b[i, 1]), float(b[i, 2]), float(b[i, 3]), float(score[keep[i]]),
                 int(cls[keep[i]])) for i in range(len(keep))]

    def best(self, rgb, cls, conf=0.3, iou=0.3):
        """Most confident box of class `cls` as (x1, y1, x2, y2), or None."""
        for x1, y1, x2, y2, s, c in self(rgb, conf=conf, iou=iou):
            if c == cls:
                return (x1, y1, x2, y2)
        return None
