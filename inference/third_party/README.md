# Third-party components (not vendored)

## V-JEPA 2 (backbone builders, ViT-L path)

```bash
git clone https://github.com/facebookresearch/vjepa2 third_party/vjepa2_repo
git -C third_party/vjepa2_repo checkout 204698b45b3712590f06245fbfba32d3be539812
```

`run.py` exports `VJEPA2_REPO=third_party/vjepa2_repo`; override with the environment variable. The ViT-H
path loads `facebook/vjepa2-vith-fpc64-256` from the Hugging Face Hub instead.

## HaWoR (hand tracker)

HaWoR is not included. Set it up following its README in its own environment (see `../envs/hawor_*.txt`),
at the commit we used:

```bash
git clone --recursive https://github.com/ThunderVVV/HaWoR /path/to/HaWoR
git -C /path/to/HaWoR checkout 66c7d410
cp third_party/hawor/run_hawor_headless.py /path/to/HaWoR/      # our headless runner (imports HaWoR)
git -C /path/to/HaWoR apply "$PWD/third_party/hawor/droid_slam_static_camera.patch"   # BSD-3 DROID-SLAM fix
```

Then set `HAWOR_PY` and `HAWOR_ROOT` in `config.env`.

**HaWoR's own hand detector is not used.** Our runner gives HaWoR the hands that Stage A detected, links
them into tracks, and then runs HaWoR's motion estimation, SLAM and infiller unchanged. It never imports
HaWoR's detection module (`lib/pipeline/tools.py`, which needs Ultralytics); a stand-in supplies the two
helpers HaWoR uses from it. So you can skip Ultralytics and `weights/external/detector.pt` when following
HaWoR's install instructions. The runner refuses to import Ultralytics even if it is installed.

`droid_slam_static_camera.patch` makes DROID-SLAM's backend a no-op when a static clip yields fewer than
two keyframes (instead of crashing on an empty edge list), and builds PTX for compute 8.6 so the
extensions also run on newer GPUs.

**Known HaWoR issue with static cameras:** on a clip where the camera does not move, HaWoR's metric-scale
estimation in `scripts/scripts_test_video/hawor_slam.py` can retry forever on NaN. If a run hangs after
"Estimating Metric Scale", use a clip with some camera motion, or bound that retry loop and fall back to
scale 1.0 when no keyframe has positive disparity in your local HaWoR checkout.

## WiLoR hand detector

Stage A and the hand probe use [WiLoR](https://github.com/rolpotamias/WiLoR)'s hand detector, converted to
ONNX and run with onnxruntime (`lib/utils/hand_detector.py`). The pipeline does not use the library the
detector was built with. You convert the detector yourself, once, in a separate environment that you can
delete afterwards:

```bash
# 1. WiLoR's detector (sha256 5ef3df44…0033), under WiLoR's license
wget -O /tmp/wilor_detector.pt \
  https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt

# 2. Convert it to ONNX in a throwaway environment. Ultralytics is needed only for this step.
python3 -m venv /tmp/yolo-export
/tmp/yolo-export/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/yolo-export/bin/pip install ultralytics onnx dill
/tmp/yolo-export/bin/pip uninstall -y opencv-python && /tmp/yolo-export/bin/pip install opencv-python-headless   # headless machines
/tmp/yolo-export/bin/yolo export model=/tmp/wilor_detector.pt format=onnx imgsz=512 opset=17 dynamic=True

# 3. Put it where the pipeline looks for it, and delete the throwaway environment
mkdir -p data/base_data/demo_data
mv /tmp/wilor_detector.onnx data/base_data/demo_data/wilor_detector.onnx
rm -rf /tmp/yolo-export /tmp/wilor_detector.pt
```

Keep `dynamic=True` and `imgsz=512`, which is the size the detector was trained at. With those settings
the ONNX detector gives the same boxes as the original to within a pixel. A fixed-size export changes the
boxes. The exported file is a converted copy of WiLoR's weights and stays under WiLoR's license, so don't
redistribute it.
