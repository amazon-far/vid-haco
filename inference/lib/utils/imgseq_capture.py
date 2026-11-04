"""
Drop-in cv2.VideoCapture replacement for a directory of numbered JPG frames
(0.jpg, 1.jpg, 2.jpg, ...), e.g. Record3D `raw/cam_X/rgbd/`.

WHY: the demo pipeline (Stage A/B/W/render) reads frames with cv2.VideoCapture.
For an image sequence, cv2.VideoCapture's image backend decodes JPEGs MATERIALLY
differently from cv2.imread (observed max per-pixel diff 65, ~60% pixels changed),
whereas training reads every frame with
cv2.imread(IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION). Feeding the model
VideoCapture-decoded pixels would be off-distribution vs training.

ImgSeqCapture decodes each frame with the SAME cv2.imread flags as the training
loaders, so image-sequence inference is bit-faithful to training. It mimics only the small
VideoCapture API the stages use: isOpened(), read()->(ret,bgr), release(),
get(CAP_PROP_FRAME_COUNT|FPS|POS_FRAMES|FRAME_WIDTH|FRAME_HEIGHT). Frames are read
sequentially in numeric-stem order (the stages never seek); returns BGR like
cv2.VideoCapture so the stages' own cvtColor(BGR2RGB) is unchanged.

fps: taken from the sibling Record3D `metadata` json ({"fps":30,...}) if present,
else 30.0 (cv2's image backend wrongly reports 25).
"""
import os
import json
import cv2

_IMREAD = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


class ImgSeqCapture:
    def __init__(self, d):
        self.dir = d
        pairs = []
        for f in os.listdir(d):
            if not f.lower().endswith(".jpg"):
                continue
            stem = os.path.splitext(f)[0]
            try:
                pairs.append((int(stem), f))
            except ValueError:
                continue  # ignore non-numeric names
        pairs.sort(key=lambda t: t[0])
        self.stems = [s for s, _ in pairs]
        self.files = [os.path.join(d, f) for _, f in pairs]
        self.n = len(self.files)
        self.pos = 0
        self._fps = self._read_fps(d)
        self._wh = None

    @staticmethod
    def _read_fps(d):
        meta = os.path.join(os.path.dirname(d.rstrip("/")), "metadata")
        try:
            with open(meta) as fh:
                v = float(json.load(fh).get("fps", 30.0))
                return v if v and v > 0 else 30.0
        except Exception:
            return 30.0

    def is_contiguous(self):
        """True iff stems are exactly 0..n-1 (stages treat read-order index as absolute frame idx)."""
        return self.n > 0 and self.stems == list(range(self.n))

    def isOpened(self):
        return self.n > 0

    def read(self):
        if self.pos >= self.n:
            return False, None
        img = cv2.imread(self.files[self.pos], _IMREAD)  # BGR, matches the training decode
        self.pos += 1
        return (img is not None), img

    def grab(self):
        """Advance one frame WITHOUT decoding it (cv2.VideoCapture.grab parity).

        Callers that walk a stream to reach a sub-range use grab() to skip cheaply; for an image
        sequence that means not calling imread at all, so skipping is essentially free.
        """
        if self.pos >= self.n:
            return False
        self.pos += 1
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.n)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            if self._wh is None and self.n:
                im = cv2.imread(self.files[0], _IMREAD)
                self._wh = (im.shape[1], im.shape[0]) if im is not None else (0, 0)
            if self._wh:
                return float(self._wh[0] if prop == cv2.CAP_PROP_FRAME_WIDTH else self._wh[1])
        return 0.0

    def release(self):
        pass


def open_capture(path):
    """ImgSeqCapture for a directory of numbered jpgs; real cv2.VideoCapture otherwise (mp4/mov)."""
    if os.path.isdir(path):
        return ImgSeqCapture(path)
    return cv2.VideoCapture(path)
