"""Readers for the eight datasets whose labels come from HACO's preprocessed data release.

HACO (Jung & Lee, NeurIPS 2025) publishes, for ARCTIC (egocentric), DECAF, Hi4D, InterHand2.6M,
H2O, MOW, PROX and HIC, per-sample hand boxes, 778-vertex MANO contact labels and the training
split lists (Hugging Face dataset `dqj5182/haco-data`); the images come from each dataset's
official release. https://github.com/dqj5182/HACO_RELEASE (docs/data_train.md) explains how to
download both into ./data/<NAME>/. Each class documents the files it reads.

Pool order: every reader enumerates the samples of HACO's *training* split in the order that
split's list file gives them (`len(reader)` samples, index i = i-th list entry). The paper's split
files data_processing/splits/<NAME>_{train,val}.json are integer indices into this order.

All readers share HandCropDataset's interface (see common.py); `transform` is accepted for
signature compatibility with the other readers and ignored.
"""
import json
import os
import pickle
import zlib

import cv2
import numpy as np

from .common import DATA_ROOT, HandCropDataset, Target, load_contact, load_npz_key


def _read_json(path):
    with open(path) as f:
        return json.load(f)


class _HacoLabelReader(HandCropDataset):
    """Shared paths: data/<NAME>/{splits/train.json, preprocessed_data/train/...}."""
    NAME = None

    def __init__(self, data_split='train', input_size=None):
        super().__init__(data_split, input_size)
        self.root_path = os.path.join(DATA_ROOT, self.NAME)
        self.data_dir = os.path.join(DATA_ROOT, self.NAME, 'data')
        labels = os.path.join(self.root_path, 'preprocessed_data', 'train')
        self.annot_data_path = os.path.join(labels, 'annot_data')
        self.contact_data_path = os.path.join(labels, 'contact_data')
        self.split_file = os.path.join(self.root_path, 'splits', 'train.json')

    def _box(self, sid, key, allow_pickle=False):
        path = os.path.join(self.annot_data_path, sid + '.npz')
        return np.asarray(load_npz_key(path, key, allow_pickle), dtype=np.float32)

    def _contact(self, sid):
        return load_contact(os.path.join(self.contact_data_path, sid + '.npy'))


class ARCTIC(_HacoLabelReader):
    """ARCTIC, egocentric camera 0 (protocol p2).

    data/ARCTIC/splits/train.json          pool: integer indices into `imgnames`
    data/ARCTIC/data/splits/p2_train.npy   pickled dict; only its 'imgnames' list is read, entries
                                           './arctic_data/data/images/<subj>/<seq>/<cam>/<frame>.jpg'
    data/ARCTIC/data/cropped_images/<subj>/<seq>/<cam>/<frame:05d>.jpg
    preprocessed_data/train/annot_data/<sid>.npz    'bbox_hand_r' (x, y, w, h), right hand
    preprocessed_data/train/contact_data/<sid>.npy  (778,)
    sid = '<subj>-<seq>-<cam>-<frame>'
    """
    NAME = 'ARCTIC'
    frame_digits, frame_ext = 5, '.jpg'
    IMAGE_PREFIX = './arctic_data/data/images/'

    def __init__(self, transform=None, data_split='train', input_size=None):
        super().__init__(data_split, input_size)
        self.split = [int(v) for v in _read_json(self.split_file)]
        db = np.load(os.path.join(self.data_dir, 'splits', 'p2_train.npy'), allow_pickle=True)
        self.imgnames = db.item()['imgnames']

    def _parts(self, i):
        rel = str(self.imgnames[self.split[i]]).rpartition(self.IMAGE_PREFIX)[2]
        subj, seq, cam, fname = rel.split('/')[:4]
        return subj, seq, cam, fname.partition('.jpg')[0]

    def sample_id(self, i):
        return '-'.join(self._parts(i))

    def target(self, i):
        subj, seq, cam, frame = self._parts(i)
        sid = f'{subj}-{seq}-{cam}-{frame}'
        seq_dir = os.path.join(self.data_dir, 'cropped_images', subj, seq, cam)
        return Target(image=self.frame_path(seq_dir, int(frame)), frame=int(frame),
                      box=self._box(sid, 'bbox_hand_r'), contact=self._contact(sid))


class Decaf(_HacoLabelReader):
    """DECAF (hand-face contact).

    data/Decaf/splits/train.json                      pool: sample ids '<seq>_<cam>_<frame>'
    data/Decaf/data/train/images/<seq>/<cam>/<frame:05d>.jpg
    preprocessed_data/train/db_bb/<sid>.npy           right-hand box (x0, y0, x1, y1)
    preprocessed_data/train/db_contact/<sid>.npy      (778,)
    """
    NAME = 'Decaf'
    frame_digits, frame_ext = 5, '.jpg'

    def __init__(self, transform=None, data_split='train', input_size=None):
        super().__init__(data_split, input_size)
        self.split = list(_read_json(self.split_file))
        labels = os.path.join(self.root_path, 'preprocessed_data', 'train')
        self.box_path = os.path.join(labels, 'db_bb')
        self.contact_data_path = os.path.join(labels, 'db_contact')

    def sample_id(self, i):
        return str(self.split[i])

    def target(self, i):
        sid = self.sample_id(i)
        seq, cam, frame = sid.split('_')
        x0, y0, x1, y1 = np.load(os.path.join(self.box_path, sid + '.npy')).tolist()
        seq_dir = os.path.join(self.data_dir, 'train', 'images', seq, cam)
        return Target(image=self.frame_path(seq_dir, int(frame)), frame=int(frame),
                      box=np.array([x0, y0, x1 - x0, y1 - y0], np.float32),
                      contact=self._contact(sid))


class Hi4D(_HacoLabelReader):
    """Hi4D (hand-body contact between two people); one sample per (frame, person).

    data/Hi4D/splits/train.json       pool: frame paths '.../<pair>/<action>/images/<cam>/<frame>.jpg'
    data/Hi4D/splits/train_pid.json   person index (0 or 1) of each pool entry
    data/Hi4D/data/<pair>/<action>/images/<cam>/<frame:06d>.jpg
    preprocessed_data/train/annot_data/<sid>.npz   'bbox_hand_r_<pid>', 'mano_r_contact_<pid>'
    sid = '<pair>-<action>-<cam>-<frame>-<pid>'
    """
    NAME = 'Hi4D'
    frame_digits, frame_ext = 6, '.jpg'

    def __init__(self, transform=None, data_split='train', input_size=None):
        super().__init__(data_split, input_size)
        self.split = list(_read_json(self.split_file))
        self.person = list(_read_json(os.path.join(self.root_path, 'splits', 'train_pid.json')))

    def _parts(self, i):
        p = self.split[i].split('/')
        return p[-5], p[-4], p[-2], p[-1].partition('.jpg')[0], self.person[i]

    def sample_id(self, i):
        pair, action, cam, frame, pid = self._parts(i)
        return f'{pair}-{action}-{cam}-{frame}-{pid}'

    def target(self, i):
        pair, action, cam, frame, pid = self._parts(i)
        with np.load(os.path.join(self.annot_data_path, self.sample_id(i) + '.npz')) as z:
            box = np.asarray(z[f'bbox_hand_r_{int(pid)}'], np.float32)
            contact = z[f'mano_r_contact_{int(pid)}'].astype(np.float32)
        seq_dir = os.path.join(self.data_dir, pair, action, 'images', cam)
        return Target(image=self.frame_path(seq_dir, int(frame)), frame=int(frame),
                      box=box, contact=contact)


class H2O(_HacoLabelReader):
    """H2O, egocentric camera.

    data/H2O/splits/train.json   pool: '<subject>/<seq>/<obj>/<cam>/rgb/<frame>.png'
    data/H2O/data/<subject>_ego/<seq>/<obj>/<cam>/rgb/<frame:06d>.png
    preprocessed_data/train/annot_data/<sid>.npz    'bbox_hand_r'
    preprocessed_data/train/contact_data/<sid>.npy  (778,)
    sid = '<subject>_ego-<seq>-<obj>-<cam>-<frame>'
    (H2O's cam*/mask/ folders hold depth maps, not segmentation, so H2O has no mask.)
    """
    NAME = 'H2O'
    frame_digits, frame_ext = 6, '.png'

    def __init__(self, transform=None, data_split='train', input_size=None):
        super().__init__(data_split, input_size)
        self.split = list(_read_json(self.split_file))

    def _parts(self, i):
        p = self.split[i].split('/')
        return p[0] + '_ego', p[1], p[2], p[3], p[5].partition('.png')[0]

    def sample_id(self, i):
        return '-'.join(self._parts(i))

    def target(self, i):
        subj, seq, obj, cam, frame = self._parts(i)
        sid = '-'.join((subj, seq, obj, cam, frame))
        seq_dir = os.path.join(self.data_dir, subj, seq, obj, cam, 'rgb')
        return Target(image=self.frame_path(seq_dir, int(frame)), frame=int(frame),
                      box=self._box(sid, 'bbox_hand_r'), contact=self._contact(sid))


class InterHand26M(_HacoLabelReader):
    """InterHand2.6M (hand-hand contact), 5 fps release.

    data/InterHand26M/splits/train.json   pool: annotation ids
    data/InterHand26M/annotations/train/InterHand2.6M_train_data.pkl
        zlib-compressed pickle of the COCO-style {'images': [...], 'annotations': [...]} tables
        (the layout of HACO's instructions); without it the official
        InterHand2.6M_train_data.json next to it is read instead. Only annotation id -> image_id
        and image id -> file_name are used.
    data/InterHand26M/images/train/<file_name>
    preprocessed_data/train/annot_data/<sid>.npz    'bbox_hand'
    preprocessed_data/train/contact_data/<sid>.npy  (778,)
    sid = '<image_id>'
    Frames have no fixed numbering: clips use the image's directory listing, ordered by the first
    number in each file name.
    """
    NAME = 'InterHand26M'

    def __init__(self, transform=None, data_split='train', input_size=None):
        super().__init__(data_split, input_size)
        self.split = [int(v) for v in _read_json(self.split_file)]
        self.img_dir = os.path.join(self.root_path, 'images', 'train')
        images, annotations = self._coco_tables(os.path.join(self.root_path, 'annotations', 'train'))
        wanted = set(self.split)
        self.image_of = {a['id']: a['image_id'] for a in annotations if a['id'] in wanted}
        used = set(self.image_of.values())
        self.file_of = {im['id']: im['file_name'] for im in images if im['id'] in used}

    @staticmethod
    def _coco_tables(ann_dir):
        cache = os.path.join(ann_dir, 'InterHand2.6M_train_data.pkl')
        if os.path.isfile(cache):
            with open(cache, 'rb') as f:
                tables = pickle.loads(zlib.decompress(f.read()))
        else:
            tables = _read_json(os.path.join(ann_dir, 'InterHand2.6M_train_data.json'))
        return tables['images'], tables['annotations']

    def sample_id(self, i):
        return str(self.image_of[self.split[i]])

    def target(self, i):
        image_id = self.image_of[self.split[i]]
        sid = str(image_id)
        return Target(image=os.path.join(self.img_dir, self.file_of[image_id]),
                      box=self._box(sid, 'bbox_hand'), contact=self._contact(sid))


def foreground_box(fg, grow):
    """Bounding box (x, y, w, h) of the True pixels of `fg`, grown by grow/2 of its width and
    height on every side (integer-truncated, clamped to the image)."""
    ys, xs = np.nonzero(fg)
    if xs.size == 0:
        raise ValueError('empty foreground mask')
    img_h, img_w = fg.shape
    box_w, box_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    left = max(0, int(xs.min() - box_w * grow / 2))
    top = max(0, int(ys.min() - box_h * grow / 2))
    right = min(img_w - 1, int(xs.max() + box_w * grow / 2))
    bottom = min(img_h - 1, int(ys.max() + box_h * grow / 2))
    return np.array([left, top, right - left + 1, bottom - top + 1], np.float32)


class MOW(_HacoLabelReader):
    """MOW (in-the-wild hand-object photos; single images, no video).

    data/MOW/splits/train.json                      pool: image ids
    data/MOW/data/images/<id>.jpg
    data/MOW/data/masks/both/<id>.jpg               hand+object foreground (HACO release), for the box
    preprocessed_data/train/contact_data/<id>.npy   (778,)
    sid = '<id>'
    The crop box is the foreground's bounding box grown by 0.65x its size on every side
    (foreground = first channel > 128). No segmentation mask is supplied, as in the released
    training code.
    """
    NAME = 'MOW'
    BOX_GROW = 1.3

    def __init__(self, transform=None, data_split='train', input_size=None):
        super().__init__(data_split, input_size)
        self.split = list(_read_json(self.split_file))
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.mask_dir = os.path.join(self.data_dir, 'masks', 'both')

    def sample_id(self, i):
        return str(self.split[i])

    def target(self, i):
        sid = self.sample_id(i)
        mask_path = os.path.join(self.mask_dir, sid + '.jpg')
        mask = cv2.imread(mask_path)
        if mask is None:
            raise IOError(f'cannot read mask: {mask_path}')
        fg = mask[:, :, 0] > 128
        return Target(image=os.path.join(self.img_dir, sid + '.jpg'),
                      box=foreground_box(fg, self.BOX_GROW), contact=self._contact(sid))


class PROX(_HacoLabelReader):
    """PROX quantitative (hand-scene contact). Frames are mirrored left-right before cropping;
    the stored boxes refer to the mirrored frame.

    data/PROX/splits/train.json   pool: frame paths '.../<seq>/Color/<frame>' (no extension)
    data/PROX/data/quantitative/recordings/<seq>/Color/<frame>.jpg
    preprocessed_data/train/annot_data/<sid>.npz    'bbox_ho'
    preprocessed_data/train/contact_data/<sid>.npy  (778,)
    sid = '<seq>-<frame>'
    Clips use the Color/ directory listing, ordered by the first number in each file name.
    """
    NAME = 'PROX'

    def __init__(self, transform=None, data_split='train', input_size=None):
        super().__init__(data_split, input_size)
        self.split = list(_read_json(self.split_file))
        self.recordings = os.path.join(self.data_dir, 'quantitative', 'recordings')

    def _parts(self, i):
        p = self.split[i].split('/')
        return p[-3], p[-1]

    def sample_id(self, i):
        return '-'.join(self._parts(i))

    def target(self, i):
        seq, frame = self._parts(i)
        sid = f'{seq}-{frame}'
        return Target(image=os.path.join(self.recordings, seq, 'Color', frame + '.jpg'),
                      box=self._box(sid, 'bbox_ho'), contact=self._contact(sid), mirror=True)


class HIC(_HacoLabelReader):
    """HIC (hand-hand / hand-object); used as single images.

    data/HIC/splits/train.json   COCO-style {'images': [...], 'annotations': [...]}; entry i of
                                 both lists is pool sample i
    data/HIC/data/<images[i]['file_name']>
    preprocessed_data/train/annot_data/<sid>.npz    'bbox_hand'
    preprocessed_data/train/contact_data/<sid>.npy  (778,)
    sid = '<annotations[i]["image_id"]>'
    """
    NAME = 'HIC'

    def __init__(self, transform=None, data_split='train', input_size=None):
        super().__init__(data_split, input_size)
        tables = _read_json(self.split_file)
        self.split = [im['file_name'] for im in tables['images']]
        self.image_ids = [a['image_id'] for a in tables['annotations']]

    def sample_id(self, i):
        return str(self.image_ids[i])

    def target(self, i):
        sid = self.sample_id(i)
        return Target(image=os.path.join(self.data_dir, self.split[i]),
                      box=self._box(sid, 'bbox_hand', allow_pickle=True), contact=self._contact(sid))


READERS = {cls.NAME: cls for cls in (ARCTIC, Decaf, Hi4D, H2O, InterHand26M, MOW, PROX, HIC)}
