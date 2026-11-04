"""HOT3D (Aria RGB) reader.

Frames are the upright (rot90 CCW) Aria camera-rgb stream extracted from recording.vrs by
hot3d_contact/haco_adapter/extract_frames.py; boxes, contact labels and the pool list are written by
hot3d_contact/haco_adapter/build_haco_hot3d.py:

    data/HOT3D/splits/train.json                              pool: '<seq>/<hand>/<frame>'
    data/HOT3D/data/<seq>/<frame:06d>.jpg
    data/HOT3D/preprocessed_data/train/annot_data/<sid>.npz   'bbox_hand_r' (x, y, w, h)
    data/HOT3D/preprocessed_data/train/contact_data/<sid>.npy (778,) contact @ 1 cm
    sid = '<seq>-<hand>-<frame:06d>'

There is one labelled pool; the train/val hold-out is made by multidata_vjepa.

Left hands: the model is right-hand-centric, so a left-hand sample mirrors the whole frame (the
left hand then looks like a right hand) and its stored box is already in mirrored coordinates. The
778-vertex contact vector is used as is, since MANO's left and right meshes share vertex order.
"""
import json
import os

import numpy as np

from ..common import DATA_ROOT, HandCropDataset, Target, load_contact, load_npz_key


class HOT3D(HandCropDataset):
    frame_digits, frame_ext = 6, '.jpg'

    def __init__(self, transform, data_split, input_size=None):
        """`transform` is ignored (kept for the reader signature); `data_split` only selects the
        crop ('train' -> random augmentation, else deterministic)."""
        super().__init__(data_split, input_size)
        self.root_path = os.path.join(DATA_ROOT, 'HOT3D')
        self.data_dir = os.path.join(DATA_ROOT, 'HOT3D', 'data')   # <seq>/<frame:06d>.jpg
        labels = os.path.join(self.root_path, 'preprocessed_data', 'train')
        self.annot_data_path = os.path.join(labels, 'annot_data')
        self.contact_data_path = os.path.join(labels, 'contact_data')
        with open(os.path.join(self.root_path, 'splits', 'train.json')) as f:
            self.split = list(json.load(f))

    def _parts(self, i):
        seq, hand, frame = self.split[i].split('/')
        return seq, hand, int(frame)

    def sample_id(self, i):
        seq, hand, frame = self._parts(i)
        return f'{seq}-{hand}-{frame:06d}'

    def target(self, i):
        seq, hand, frame = self._parts(i)
        sid = f'{seq}-{hand}-{frame:06d}'
        box = load_npz_key(os.path.join(self.annot_data_path, sid + '.npz'), 'bbox_hand_r',
                           allow_pickle=True)
        return Target(image=self.frame_path(os.path.join(self.data_dir, seq), frame), frame=frame,
                      box=np.asarray(box, np.float32), mirror=(hand == 'left'),
                      contact=load_contact(os.path.join(self.contact_data_path, sid + '.npy')))
