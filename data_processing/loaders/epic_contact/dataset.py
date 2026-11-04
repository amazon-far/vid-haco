"""EPIC-Contact (Sid2697/epic-contact) reader: the manually annotated central frame of each
EPIC-Grasps stable-grasp clip.

    data/epic_contact/splits/{train,val}.json        official clip lists (val = official test)
    data/epic_contact/data/images/<clip>.jpg          central frame (native ~456 x 256)
    data/epic_contact/preprocessed_data/<sub>/annot_data/<clip>.npz
        bbox_hand, bbox_ho, bbox_obj, annotated_side ('l' / 'r'), img_w, img_h
    data/epic_contact/preprocessed_data/<sub>/contact_data/<clip>.npy
        (778,) binary MANO contact: the union of the sparse finger / thumb / palm annotation
        (contact_vertices_sparse.json) mapped to the 778-vertex topology
    <sub> = the official list ('train' or 'val') that contains the clip

The hand box is the posed hand_mesh.obj projected with the dataset's fixed pinhole camera (f = 5000,
principal point at the image centre). The conversion notes describe the stored boxes as
(x0, y0, x1, y1); the crop reads `bbox_hand` as (x, y, w, h), which is how the paper's runs used it.

Pool = every clip of both official lists, sorted by name: the same 2,272 entries in train and val
mode, so multidata_vjepa can carve its train/val split by index (val = the 237 official test clips).
sid = '<clip>'.
"""
import json
import os

import numpy as np

from ..common import DATA_ROOT, HandCropDataset, Target, load_contact, load_npz_key


class epic_contact(HandCropDataset):

    def __init__(self, transform, data_split, input_size=None):
        """`transform` is ignored (kept for the reader signature); `data_split` only selects the
        crop ('train' -> random augmentation, else deterministic)."""
        super().__init__('train' if data_split == 'train' else 'val', input_size)
        self.root_path = os.path.join(DATA_ROOT, 'epic_contact')
        self.img_dir = os.path.join(self.root_path, 'data', 'images')
        self.preproc_root = os.path.join(self.root_path, 'preprocessed_data')
        self._subdir, clips = {}, []
        for sub in ('train', 'val'):
            with open(os.path.join(self.root_path, 'splits', f'{sub}.json')) as f:
                listed = json.load(f)
            clips += listed
            self._subdir.update((clip, sub) for clip in listed)
        self.split = sorted(clips)

    def sample_id(self, i):
        return self.split[i]

    def target(self, i):
        clip = self.split[i]
        labels = os.path.join(self.preproc_root, self._subdir[clip])
        box = load_npz_key(os.path.join(labels, 'annot_data', clip + '.npz'), 'bbox_hand',
                           allow_pickle=True)
        return Target(image=os.path.join(self.img_dir, clip + '.jpg'),
                      box=np.asarray(box, np.float32),
                      contact=load_contact(os.path.join(labels, 'contact_data', clip + '.npy')))
