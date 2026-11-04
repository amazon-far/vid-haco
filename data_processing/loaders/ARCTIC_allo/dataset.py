"""ARCTIC allocentric-camera reader (cameras 1..8, protocol p1).

ARCTIC has 9 cameras: camera 0 is egocentric (protocol p2, the `ARCTIC` reader) and cameras 1-8 are
static allocentric views. This dataset adds those views without touching `ARCTIC`.

Shared with ARCTIC (nothing duplicated):
  images   data/ARCTIC/data/cropped_images/<subj>/<seq>/<cam>/<frame:05d>.jpg   (1000 x 1000)
  contact  data/ARCTIC/preprocessed_data/train/contact_data/<subj>-<seq>-0-<frame:05d>.npy
           The 778-vertex contact map is a property of the 3D hand-object geometry, hence the same
           for every camera: the camera-0 label is used for all views.

Per view there is only the hand box, derived with ARCTIC's crop transform from the 21 right-hand
2D joints and expanded 2x (arctic_allo/build_arctic_allo.py). Everything is in one array file:

  data/ARCTIC_allo/index.npz   seqs (S,) '<subj>/<seq>' | seq_ids (N,) | cams (N,) | frames (N,)
                               | bbox (N, 4) float32 (x, y, w, h) in cropped-image pixels
  pool order = row order of index.npz;  sid = '<subj>-<seq>-<cam>-<frame:05d>'
"""
import os

import numpy as np

from ..common import DATA_ROOT, HandCropDataset, Target, load_contact


class ARCTIC_allo(HandCropDataset):
    frame_digits, frame_ext = 5, '.jpg'

    def __init__(self, transform, data_split, input_size=None):
        """`transform` is ignored (kept for the reader signature); `data_split` only selects the
        crop ('train' -> random augmentation, else deterministic)."""
        super().__init__(data_split, input_size)
        self.root_path = os.path.join(DATA_ROOT, 'ARCTIC_allo')
        self.data_dir = os.path.join(DATA_ROOT, 'ARCTIC', 'data', 'cropped_images')
        self.contact_data_path = os.path.join(DATA_ROOT, 'ARCTIC', 'preprocessed_data', 'train',
                                              'contact_data')
        self.annot_data_path = self.contact_data_path   # boxes live in index.npz
        with np.load(os.path.join(self.root_path, 'index.npz'), allow_pickle=True) as index:
            self.seqs = [str(s) for s in index['seqs']]
            self.seq_ids = index['seq_ids']
            self.cams = index['cams']
            self.frames = index['frames']
            self.bboxes = index['bbox']
        self.split = self.frames

    def _parts(self, i):
        subj, seq = self.seqs[int(self.seq_ids[i])].split('/')
        return subj, seq, int(self.cams[i]), int(self.frames[i])

    def sample_id(self, i):
        subj, seq, cam, frame = self._parts(i)
        return f'{subj}-{seq}-{cam}-{frame:05d}'

    def target(self, i):
        subj, seq, cam, frame = self._parts(i)
        label = os.path.join(self.contact_data_path, f'{subj}-{seq}-0-{frame:05d}.npy')
        seq_dir = os.path.join(self.data_dir, subj, seq, str(cam))
        return Target(image=self.frame_path(seq_dir, frame), frame=frame,
                      box=np.asarray(self.bboxes[i], np.float32), contact=load_contact(label))
