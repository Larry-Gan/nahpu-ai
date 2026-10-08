"""Run PaddleOCR 3.7.0 on the OCR fixtures with the pinned PP-OCRv6 small ONNX files.

Route: the real PaddleOCR pipeline (paddleocr.PaddleOCR -> paddlex OCR pipeline)
with engine="onnxruntime", so every pre/post-processing step is PaddleX's own code
and inference runs on the exact ONNX files in NAHPU_MODEL_DIR (default <repo>/models).
PaddlePaddle itself is not needed and no model is downloaded.

Writes into rust/crates/nahpu_ocr_burn/tests/data:
  <name>.reference.json        for each synthetic fixture: detected lines in
                               pipeline order, with all settings
  neon/<photo>.reference.json  the same for each NEON photo, which
                               scripts/fetch-model.sh puts beside the models

With image paths as arguments, writes <stem>.reference.json next to each image
instead (no expected-text check).

Run from the repo root after scripts/fetch-model.sh, in a venv with
scripts/ocr/requirements.txt installed:
    scripts/ocr/.venv/bin/python -I scripts/ocr/make_references.py

requirements.txt pins the packages named here; requirements.lock.txt lists
every package in the environment that made the committed references.

Recognition runs one crop at a time (batch size 1) to match the Rust port. The
pipeline default of 6 pads each crop to the widest in its batch, which moved
scores by up to 0.025 on these fixtures (0.022 on form.png).
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "rust" / "crates" / "nahpu_ocr_burn" / "tests" / "data"
FIXTURES = ["label", "form", "two-column", "vertical"]
NEON_PHOTOS = [
    "ocr01_beetle_pasimachus_sublaevis_label_closeup",
    "ocr07_mammal_blarina_brevicauda_handwritten_tag",
    "ocr08_mammal_sorex_trowbridgii_handwritten_notes",
    "ocr11_plant_hieracium_scouleri_herbarium_printed_label",
]
DET_MODEL = "PP-OCRv6_small_det"
REC_MODEL = "PP-OCRv6_small_rec"
REC_BATCH_SIZE = 1
PACKAGES = [
    "paddleocr",
    "paddlex",
    "onnxruntime",
    "numpy",
    "opencv-contrib-python",
    "pyclipper",
    "shapely",
    "pillow",
    "matplotlib",
]


def model_dir():
    return Path(os.environ.get("NAHPU_MODEL_DIR", REPO / "models"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_model_files(model_name):
    onnx = model_dir() / f"{model_name}.onnx"
    yml = model_dir() / f"{model_name}.yml"
    for path in (onnx, yml):
        if not path.is_file():
            sys.exit(f"missing model file {path}; set NAHPU_MODEL_DIR")
    return onnx, yml


def neon_photo(name):
    path = model_dir() / f"{name}.jpg"
    if not path.is_file():
        sys.exit(f"missing photo {path}; run scripts/fetch-model.sh {name}")
    return path


def stage_model_dir(root, model_name):
    """PaddleX expects a directory holding inference.onnx and inference.yml."""
    onnx, yml = pinned_model_files(model_name)
    directory = Path(root) / model_name
    directory.mkdir()
    shutil.copyfile(onnx, directory / "inference.onnx")
    shutil.copyfile(yml, directory / "inference.yml")
    return directory


def build_pipeline(root):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name=DET_MODEL,
        text_detection_model_dir=str(stage_model_dir(root, DET_MODEL)),
        text_recognition_model_name=REC_MODEL,
        text_recognition_model_dir=str(stage_model_dir(root, REC_MODEL)),
        text_recognition_batch_size=REC_BATCH_SIZE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="onnxruntime",
        device="cpu",
    )


def inner_pipeline(ocr):
    from paddlex.inference.models.runners import ONNXRuntimeRunner

    inner = ocr.paddlex_pipeline._pipeline
    assert not inner.use_doc_preprocessor
    assert not inner.use_textline_orientation
    for model in (inner.text_det_model, inner.text_rec_model):
        assert isinstance(model.runner, ONNXRuntimeRunner), type(model.runner)
    return inner


def package_versions():
    versions = {"python": platform.python_version()}
    for name in PACKAGES:
        versions[name] = importlib.metadata.version(name)
    try:
        versions["paddlepaddle"] = importlib.metadata.version("paddlepaddle")
    except importlib.metadata.PackageNotFoundError:
        versions["paddlepaddle"] = None
    return versions


def model_metadata():
    metadata = {"dir_env": "NAHPU_MODEL_DIR"}
    for key, name in (("det", DET_MODEL), ("rec", REC_MODEL)):
        onnx, yml = pinned_model_files(name)
        metadata[key] = {
            "name": name,
            "onnx": onnx.name,
            "onnx_sha256": sha256(onnx),
            "yml": yml.name,
            "yml_sha256": sha256(yml),
        }
    return metadata


def pipeline_parameters(inner):
    det = inner.text_det_model
    rec = inner.text_rec_model
    normalize = det.pre_tfs["Normalize"]
    det_ops = det.config["PreProcess"]["transform_ops"]
    det_norm_cfg = next(op["NormalizeImage"] for op in det_ops if "NormalizeImage" in op)
    post = det.post_op
    resize = rec.pre_tfs["ReisizeNorm"]
    return {
        "pipeline_image_reader_format": inner.img_reader.format,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "det": {
            "image_reader_format": det.pre_tfs["Read"].format,
            "resize": "DetResizeForTest type0",
            "limit_side_len": inner.text_det_limit_side_len,
            "limit_type": inner.text_det_limit_type,
            "max_side_limit": inner.text_det_max_side_limit,
            "resize_round": "python round() (half to even) of side/32, times 32, min 32",
            "interpolation": "cv2.INTER_LINEAR",
            "normalize_scale": det_norm_cfg["scale"],
            "normalize_mean": det_norm_cfg["mean"],
            "normalize_std": det_norm_cfg["std"],
            "normalize_order": det_norm_cfg["order"],
            "normalize_alpha": [float(a) for a in normalize.alpha],
            "normalize_beta": [float(b) for b in normalize.beta],
            "mean_std_applied_to_channels": "in image order, i.e. B,G,R",
            "layout": "NCHW",
            "thresh": inner.text_det_thresh,
            "box_thresh": inner.text_det_box_thresh,
            "unclip_ratio": inner.text_det_unclip_ratio,
            "max_candidates": post.max_candidates,
            "min_size": post.min_size,
            "score_mode": post.score_mode,
            "use_dilation": post.use_dilation,
            "box_type": post.box_type,
            "sort": "SortQuadBoxes: sort by (y0, x0), then bubble left-to-right when |dy| < 10",
        },
        "crop": {
            "method": "CropByPolys quad: minAreaRect of int32 quad, perspective warp",
            "interpolation": "cv2.INTER_CUBIC",
            "border": "cv2.BORDER_REPLICATE",
            "rotate_when_h_over_w_at_least": 1.5,
            "rotation": "np.rot90 (counter-clockwise)",
        },
        "rec": {
            "image_reader_format": rec.pre_tfs["Read"].format,
            "image_shape": list(resize.rec_image_shape),
            "max_width": resize.max_imgW,
            "interpolation": "cv2.INTER_LINEAR",
            "normalize": "x / 255, then (x - 0.5) / 0.5",
            "pad": "right, value 0.0 after normalisation",
            "batch_size": rec.batch_sampler.batch_size,
            "pipeline_default_batch_size": 6,
            "batch_size_note": "batch 6 pads crops to the widest in the batch; scores moved by up to 0.025 on these fixtures (0.022 on form.png)",
            "score_thresh": inner.text_rec_score_thresh,
            "decode": "CTC greedy: argmax per step, merge repeats, drop blank index 0",
            "score": "mean of max probability over kept steps (0 if none)",
            "dictionary_source": "PostProcess.character_dict in the rec yml",
            "dictionary_entries_in_yml": len(rec.config["PostProcess"]["character_dict"]),
            "space_appended": True,
            "blank_prepended": True,
            "class_count": len(rec.post_op.character),
        },
    }


def read_expected(name):
    return (DATA_DIR / f"{name}.expected.txt").read_text(encoding="ascii").splitlines()


def quad_key(poly):
    return tuple(map(tuple, np.asarray(poly).astype(int).tolist()))


def detection_scores(inner, image_path):
    image = inner.img_reader([str(image_path)])[0]
    *_, polys, scores = run_detection_stage(inner, image)
    return {quad_key(poly): float(score) for poly, score in zip(polys, scores)}


def run_fixture(ocr, inner, image_path):
    result = next(iter(ocr.predict(str(image_path))))
    det_scores = detection_scores(inner, image_path)
    lines = [
        {
            "text": text,
            "score": float(score),
            "det_score": det_scores[quad_key(poly)],
            "quad": np.asarray(poly).astype(int).tolist(),
        }
        for text, score, poly in zip(result["rec_texts"], result["rec_scores"], result["rec_polys"])
    ]
    return lines, result["text_det_params"]


def compare_with_expected(name, lines):
    expected = read_expected(name)
    texts = [line["text"] for line in lines]
    missing = sorted(set(expected) - set(texts))
    extra = sorted(set(texts) - set(expected))
    return {
        "expected_line_count": len(expected),
        "detected_line_count": len(texts),
        "same_set_of_lines": not missing and not extra and len(texts) == len(expected),
        "missing": missing,
        "unexpected": extra,
        "pipeline_order_equals_reading_order": texts == expected,
    }


def to_json(path, value):
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_reference(image_path, out_dir, lines, det_params, header, extra):
    image = cv2.imread(str(image_path))
    reference = {
        "fixture": image_path.name,
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        **header,
        "det_params_reported_by_pipeline": det_params,
        **extra,
        "lines": lines,
    }
    to_json(out_dir / f"{image_path.stem}.reference.json", reference)


def image_info(inner, image_path):
    from PIL import Image

    backend = inner.img_reader._img_reader._backend
    with Image.open(image_path) as image:
        orientation = image.getexif().get(0x0112)
    return {
        "image_sha256": sha256(image_path),
        "image_reader": {
            "call": f"{type(backend).__name__}: cv2.imdecode(file bytes, flags={backend.flags})",
            "applies_exif_orientation": not backend.flags & cv2.IMREAD_IGNORE_ORIENTATION,
            "exif_orientation_tag": orientation,
        },
    }


def run_detection_stage(inner, image):
    det = inner.text_det_model
    params = inner.get_text_det_params()
    raw = det.pre_tfs["Read"](imgs=[image])
    resized, shapes = det.pre_tfs["Resize"](
        imgs=raw,
        limit_side_len=params["limit_side_len"],
        limit_type=params["limit_type"],
        max_side_limit=params["max_side_limit"],
    )
    normalized = det.pre_tfs["Normalize"](imgs=resized)
    chw = det.pre_tfs["ToCHW"](imgs=normalized)
    x = det.pre_tfs["ToBatch"](imgs=chw)[0]
    preds = det.runner(x=[x])
    polys, scores = det.post_op(
        preds,
        shapes,
        thresh=params["thresh"],
        box_thresh=params["box_thresh"],
        unclip_ratio=params["unclip_ratio"],
    )
    return x, preds[0], shapes[0], polys[0], scores[0]


def main(paths):
    header = {
        "route": "paddleocr.PaddleOCR (paddlex OCR pipeline) with engine='onnxruntime' on the pinned ONNX files",
        "packages": package_versions(),
        "models": model_metadata(),
    }
    with tempfile.TemporaryDirectory() as root:
        ocr = build_pipeline(root)
        inner = inner_pipeline(ocr)
        header["params"] = pipeline_parameters(inner)
        if paths:
            photos = [(path, path.parent) for path in map(Path, paths)]
        else:
            photos = [(neon_photo(name), DATA_DIR / "neon") for name in NEON_PHOTOS]
        for path, out_dir in photos:
            lines, det_params = run_fixture(ocr, inner, path)
            write_reference(path, out_dir, lines, det_params, header, image_info(inner, path))
            print(path.name, len(lines), "lines")
        for name in [] if paths else FIXTURES:
            lines, det_params = run_fixture(ocr, inner, DATA_DIR / f"{name}.png")
            check = compare_with_expected(name, lines)
            write_reference(DATA_DIR / f"{name}.png", DATA_DIR, lines, det_params, header, {"check": check})
            print(name, json.dumps(check))


if __name__ == "__main__":
    main(sys.argv[1:])
