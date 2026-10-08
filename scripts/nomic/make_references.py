"""Compute the Nomic embed v1.5 reference vectors that the Burn tests compare against.

Runs the pinned ONNX files with ONNX Runtime and writes TSVs into
rust/crates/nahpu_embed_burn/tests/data/. Not part of the build.

Run from the repo root after scripts/fetch-model.sh, in a venv with
scripts/nomic/requirements.txt installed:
    scripts/nomic/.venv/bin/python -I scripts/nomic/make_references.py
"""

import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime as ort
import PIL
from PIL import Image, ImageOps
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("NAHPU_MODEL_DIR", REPO / "models"))
DATA = REPO / "rust/crates/nahpu_embed_burn/tests/data"

TEXT_MODEL = "nomic-embed-text-v1_5.onnx"
TEXT_TOKENIZER = "nomic-embed-text-v1_5.tokenizer.json"
TEXT_SOURCE = "nomic-ai/nomic-embed-text-v1.5 @ e9b6763023c676ca8431644204f50c2b100d9aab, onnx/model.onnx"
VISION_MODEL = "nomic-embed-vision-v1_5.onnx"
VISION_CONFIG = "nomic-embed-vision-v1_5.preprocessor_config.json"
VISION_SOURCE = "nomic-ai/nomic-embed-vision-v1.5 @ e3a725bce72db07ca4adb1d83da08903f3ee02f8, onnx/model.onnx"

SENTENCES = [
    "A small brown bird perched on a branch.",
    "A tiny sparrow sitting in a tree.",
    "Diesel engine maintenance schedule.",
]
PREFIXES = {"query": "search_query: ", "document": "search_document: "}
# NEON photos, fetched into the models directory by scripts/fetch-model.sh.
PHOTOS = [
    "emb01_beetle_chlaenius_aestivus_dorsal",
    "emb05_mosquito_aedes_monticola_lateral",
    "emb06_lizard_holbrookia_maculata_dorsal",
    "emb07_salamander_plethodon_cinereus_three",
    "emb08_toad_anaxyrus_americanus_two",
    "emb10_mammal_peromyscus_ventral",
]
MAX_TOKENS = 512
LAYER_NORM_EPS = 1e-5

IMAGE_SIZE = 224
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
MEAN = np.array(IMAGE_MEAN, dtype=np.float32)
STD = np.array(IMAGE_STD, dtype=np.float32)
RESCALE = 1 / 255

SYNTHETIC_SAMPLES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (1, 0, 0), (2, 0, 0), (2, 223, 223), (1, 100, 37)]


def check_models_present():
    needed = [TEXT_MODEL, TEXT_TOKENIZER, VISION_MODEL, VISION_CONFIG] + [f"{photo}.jpg" for photo in PHOTOS]
    missing = [name for name in needed if not (MODELS / name).is_file()]
    if missing:
        raise SystemExit(f"missing in {MODELS}: {', '.join(missing)} (set NAHPU_MODEL_DIR or run scripts/fetch-model.sh)")


def check_preprocessor_config():
    config = json.loads((MODELS / VISION_CONFIG).read_text(encoding="utf-8"))
    square = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    expected = {
        "image_processor_type": "CLIPImageProcessor",
        "do_convert_rgb": True,
        "do_resize": True,
        "size": square,
        "resample": int(Image.Resampling.BICUBIC),
        "do_center_crop": True,
        "crop_size": square,
        "do_rescale": True,
        "rescale_factor": RESCALE,
        "do_normalize": True,
        "image_mean": IMAGE_MEAN,
        "image_std": IMAGE_STD,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise SystemExit(f"{VISION_CONFIG}: {key} is {config[key]!r}, this script assumes {value!r}")


def session(name):
    return ort.InferenceSession(str(MODELS / name), providers=["CPUExecutionProvider"])


def format_vector(vector):
    return ",".join(f"{x:.6f}" for x in vector)


def unit_length(vector):
    return vector / np.linalg.norm(vector)


def layer_norm_without_weights(vector):
    centred = vector - vector.mean()
    return centred / np.sqrt(centred.var() + LAYER_NORM_EPS)


def embed_text(model, tokenizer, role, sentence):
    encoding = tokenizer.encode(PREFIXES[role] + sentence)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    mask = np.array([encoding.attention_mask], dtype=np.int64)
    hidden = model.run(
        ["last_hidden_state"],
        {"input_ids": input_ids, "token_type_ids": np.zeros_like(input_ids), "attention_mask": mask},
    )[0][0].astype(np.float64)
    weights = mask[0].astype(np.float64)[:, None]
    pooled = (hidden * weights).sum(axis=0) / weights.sum()
    return unit_length(layer_norm_without_weights(pooled))


def load_tokenizer():
    tokenizer = Tokenizer.from_file(str(MODELS / TEXT_TOKENIZER))
    tokenizer.enable_truncation(MAX_TOKENS)
    tokenizer.no_padding()
    return tokenizer


def normalise_pixels(pixels_hwc_unit):
    return ((pixels_hwc_unit - MEAN) / STD).transpose(2, 0, 1)


def preprocess_image(path):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    unit = (np.asarray(resized, dtype=np.float64) * RESCALE).astype(np.float32)
    return normalise_pixels(unit)[None], image.size


def synthetic_pixel_values():
    c, y, x = np.mgrid[0:3, 0:IMAGE_SIZE, 0:IMAGE_SIZE]
    byte = (x * 7 + y * 13 + c * 29) % 256
    unit = byte.astype(np.float32) / np.float32(255)
    return ((unit - MEAN[:, None, None]) / STD[:, None, None])[None].astype(np.float32)


def embed_pixels(model, pixel_values):
    hidden = model.run(["last_hidden_state"], {"pixel_values": pixel_values})[0]
    return unit_length(hidden[0, 0].astype(np.float64))


def write_lines(name, lines):
    path = DATA / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(REPO)}")


def text_header():
    return [
        f"# nomic-embed-text-v1.5 embeddings from ONNX Runtime {ort.__version__}, {TEXT_SOURCE}",
        f"# recipe: prefix 'search_query: ' (query) or 'search_document: ' (document); tokenizer.json with truncation to {MAX_TOKENS} tokens, no padding",
        "# inputs: input_ids int64 [1,n], token_type_ids int64 all zeros, attention_mask int64 all ones; output last_hidden_state [1,n,768]",
        f"# mean-pool over the attention mask; layer norm without weights ((x - mean) / sqrt(biased var + {LAYER_NORM_EPS})); L2-normalise",
        "# post-processing done in float64; columns: role, sentence, 768 comma-separated floats",
    ]


def write_text_references(model, tokenizer):
    rows = []
    for sentence in SENTENCES:
        for role in PREFIXES:
            vector = embed_text(model, tokenizer, role, sentence)
            rows.append(f"{role}\t{sentence}\t{format_vector(vector)}")
    write_lines("nomic-text-reference.tsv", text_header() + rows)


def write_vision_tensor_reference(model):
    pixel_values = synthetic_pixel_values()
    header = [
        f"# nomic-embed-vision-v1.5 embedding of a synthetic tensor, ONNX Runtime {ort.__version__}, {VISION_SOURCE}",
        "# pixel_values [1,3,224,224] float32, for c in 0..3, y in 0..224, x in 0..224:",
        "#   byte = (x*7 + y*13 + c*29) % 256 (integer)",
        "#   value = ((byte as f32) / 255.0_f32 - MEAN[c]) / STD[c], each step one f32 operation",
        "#   MEAN = [0.48145466_f32, 0.4578275_f32, 0.40821073_f32], STD = [0.26862954_f32, 0.26130258_f32, 0.27577711_f32]",
        "# output: token 0 of last_hidden_state [1,197,768], L2-normalised in float64",
        "# rows: 'input' c y x value (shortest round-trip f32 text, parse as f32 for an exact match); 'embedding' 768 floats",
    ]
    inputs = [
        f"input\t{c}\t{y}\t{x}\t{np.format_float_positional(pixel_values[0, c, y, x], unique=True)}"
        for c, y, x in SYNTHETIC_SAMPLES
    ]
    vector = embed_pixels(model, pixel_values)
    write_lines("nomic-vision-tensor-reference.tsv", header + inputs + [f"embedding\t{format_vector(vector)}"])


def image_header():
    return [
        f"# nomic-embed-vision-v1.5 embeddings of the NEON photos from scripts/fetch-model.sh, ONNX Runtime {ort.__version__}, {VISION_SOURCE}",
        f"# preprocessing per preprocessor_config.json with Pillow {PIL.__version__}: open, ImageOps.exif_transpose, convert('RGB'),",
        "#   resize to 224x224 ignoring aspect ratio with Image.Resampling.BICUBIC (resample=3; Pillow bicubic a=-0.5, antialiased when downscaling),",
        "#   center crop 224x224 is then a no-op, uint8 -> float64 * (1/255) -> float32, (x - mean) / std in float32, HWC -> CHW",
        "# output: token 0 of last_hidden_state [1,197,768], L2-normalised in float64",
        "# columns: photo, displayed width x height after EXIF orientation, preprocessed tensor mean per channel (R,G,B), 768 floats",
    ]


def write_image_references(model):
    rows = []
    for photo in PHOTOS:
        pixel_values, (width, height) = preprocess_image(MODELS / f"{photo}.jpg")
        vector = embed_pixels(model, pixel_values)
        channel_means = ",".join(f"{m:.6f}" for m in pixel_values[0].mean(axis=(1, 2), dtype=np.float64))
        rows.append(f"{photo}\t{width}x{height}\t{channel_means}\t{format_vector(vector)}")
    write_lines("nomic-vision-image-reference.tsv", image_header() + rows)


def main():
    print(f"models from {MODELS}; onnxruntime {ort.__version__}, tokenizers {version('tokenizers')}, pillow {PIL.__version__}, numpy {np.__version__}")
    check_models_present()
    text_model, vision_model = session(TEXT_MODEL), session(VISION_MODEL)
    tokenizer = load_tokenizer()
    check_preprocessor_config()
    write_text_references(text_model, tokenizer)
    write_vision_tensor_reference(vision_model)
    write_image_references(vision_model)


if __name__ == "__main__":
    main()
