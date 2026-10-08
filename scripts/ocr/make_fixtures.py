"""Render the synthetic OCR fixtures and their expected text.

Writes <name>.png and <name>.expected.txt into rust/crates/nahpu_ocr_burn/tests/data.
The expected file lists every rendered text run in reading order, one per line.

Font: DejaVu Sans and DejaVu Sans Bold as shipped in matplotlib's package data
(matplotlib pinned in requirements.txt; licence: Bitstream Vera / DejaVu
public-domain-style licence, see mpl-data/fonts/ttf/LICENSE_DEJAVU).
The font files are checked by sha256 so the images cannot silently change.

Run from the repo root, in a venv with scripts/ocr/requirements.txt installed:
    scripts/ocr/.venv/bin/python -I scripts/ocr/make_fixtures.py
"""

import hashlib
import sys
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "rust" / "crates" / "nahpu_ocr_burn" / "tests" / "data"

FONT_SHA256 = {
    "DejaVuSans.ttf": "3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22",
    "DejaVuSans-Bold.ttf": "b184b89e3c1075f22f6b71575b6fc20d4972b3cfd3b23322ca6fd596dcaef167",
}

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE_INK = (20, 40, 120)
CREAM_PAPER = (250, 245, 230)


class Run(NamedTuple):
    x: int
    y: int
    text: str
    font: ImageFont.FreeTypeFont
    rotated_clockwise: bool = False


def font_path(file_name):
    try:
        import matplotlib
    except ImportError:
        sys.exit("matplotlib is required: it provides the pinned DejaVu Sans font files")
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / file_name
    if not path.is_file():
        sys.exit(f"font not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = FONT_SHA256[file_name]
    if digest != expected:
        sys.exit(f"font {path} has sha256 {digest}, expected {expected}")
    return path


def load_font(file_name, size):
    return ImageFont.truetype(str(font_path(file_name)), size)


def rotated_text(run, ink, paper):
    left, top, right, bottom = run.font.getbbox(run.text)
    strip = Image.new("RGB", (right, bottom), paper)
    ImageDraw.Draw(strip).text((0, 0), run.text, font=run.font, fill=ink)
    return strip.transpose(Image.Transpose.ROTATE_270)


def render(size, runs, ink, paper):
    image = Image.new("RGB", size, paper)
    draw = ImageDraw.Draw(image)
    for run in runs:
        if run.rotated_clockwise:
            image.paste(rotated_text(run, ink, paper), (run.x, run.y))
        else:
            draw.text((run.x, run.y), run.text, font=run.font, fill=ink)
    return image


def label_runs():
    regular = load_font("DejaVuSans.ttf", 34)
    bold = load_font("DejaVuSans-Bold.ttf", 38)
    lines = [
        (bold, "Rattus tiomanicus"),
        (regular, "Malaysia, Sabah, Mt. Kinabalu"),
        (regular, "Elevation 1520 m"),
        (regular, "Date: 14 March 1998"),
        (regular, "Collector: R. Hartono"),
        (regular, "Cat. No. FMNH 208711"),
    ]
    return [Run(70, 60 + index * 70, text, font) for index, (font, text) in enumerate(lines)]


def form_runs():
    font = load_font("DejaVuSans.ttf", 30)
    rows = [
        ("Catalogue No:", "NHM 47291"),
        ("Species:", "Sundamys muelleri"),
        ("Sex:", "Male"),
        ("Locality:", "Kinabalu Park, Sabah, near Layang Layang"),
        ("Date:", "12 June 2019"),
        ("Collector:", "A. Tan"),
        ("Weight:", "84 g"),
        ("Preparation:", "Skin and skull"),
    ]
    runs = []
    for index, (key, value) in enumerate(rows):
        y = 70 + index * 72
        runs.append(Run(60, y, key, font))
        runs.append(Run(420, y, value, font))
    return runs


LEFT_COLUMN = [
    "We reached the ridge camp",
    "late in the afternoon after",
    "a long climb through mossy",
    "forest. The trail was wet",
    "and the rain did not stop",
    "until dusk. Four traps were",
    "set along the stream and",
    "six more near the old hut.",
    "Two rats were taken before",
    "midnight and kept in bags.",
]

RIGHT_COLUMN = [
    "In the morning we checked",
    "every trap and found three",
    "shrews and one squirrel.",
    "The team measured each one",
    "and wrote the notes by the",
    "fire. Clouds came down over",
    "the valley at noon, so we",
    "packed up and walked back.",
    "The next camp is lower.",
]


def two_column_runs():
    title_font = load_font("DejaVuSans-Bold.ttf", 40)
    body_font = load_font("DejaVuSans.ttf", 28)
    runs = [Run(60, 60, "Field Notes from the Crocker Range", title_font)]
    for column_x, lines in ((60, LEFT_COLUMN), (660, RIGHT_COLUMN)):
        for index, text in enumerate(lines):
            runs.append(Run(column_x, 170 + index * 60, text, body_font))
    return runs


def vertical_runs():
    font = load_font("DejaVuSans.ttf", 32)
    lines = ["Drawer 14, Rodentia", "Muridae from Borneo", "Sorted by locality"]
    runs = [Run(60, 70 + index * 70, text, font) for index, text in enumerate(lines)]
    runs.append(Run(700, 40, "Checked in May 2021", font, rotated_clockwise=True))
    return runs


FIXTURES = {
    "label": ((800, 512), label_runs, BLUE_INK, CREAM_PAPER),
    "form": ((1168, 680), form_runs, BLACK, WHITE),
    "two-column": ((1200, 800), two_column_runs, BLACK, WHITE),
    "vertical": ((800, 448), vertical_runs, BLACK, WHITE),
}


def write_fixture(name, size, runs, ink, paper):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render(size, runs, ink, paper).save(OUT_DIR / f"{name}.png", optimize=True)
    expected = "".join(run.text + "\n" for run in runs)
    (OUT_DIR / f"{name}.expected.txt").write_text(expected, encoding="ascii", newline="\n")
    print(f"{name}: {size[0]}x{size[1]}, {len(runs)} lines")


def main():
    for name, (size, make_runs, ink, paper) in FIXTURES.items():
        write_fixture(name, size, make_runs(), ink, paper)


if __name__ == "__main__":
    main()
