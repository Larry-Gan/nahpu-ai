use std::collections::HashSet;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, ImageEncoder, ImageFormat, Rgb, RgbImage};
use nahpu_ocr_burn::{OcrEngine, Quad, TextLine, HAS_OCR};

const FIXTURES: [&str; 4] = ["label", "form", "two-column", "vertical"];
/// Fetched by `scripts/fetch-model.sh`, not committed.
const NEON_PHOTOS: [&str; 4] = [
    "ocr08_mammal_sorex_trowbridgii_handwritten_notes",
    "ocr11_plant_hieracium_scouleri_herbarium_printed_label",
    "ocr01_beetle_pasimachus_sublaevis_label_closeup",
    "ocr07_mammal_blarina_brevicauda_handwritten_tag",
];
/// The highest character error rate a photo may have against PaddleOCR.
const MAX_ERROR_RATE: f32 = 0.05;
/// PaddleOCR's lines scored below this are left out of the comparison, with
/// ours nearest to them. On ocr07 that drops the ruler's tick marks (read as
/// gibberish at about 0.4) and the sideways "neon" logo on the tag (read as
/// "Ueeu", 0.43); on the other three photos it drops nothing.
const MIN_SCORE: f32 = 0.5;
const SKIPPED: &str = "skipped: PP-OCRv6 not compiled in — run scripts/fetch-model.sh PP-OCRv6_small_det PP-OCRv6_small_rec, then rebuild";

/// What PaddleOCR 3.7.0 read on the same page: the same lines, the same text,
/// and every box corner within a few pixels.
#[test]
fn ocr_fixtures_match_paddleocr() {
    let Some(engine) = load_engine() else {
        eprintln!("{SKIPPED}");
        return;
    };

    for fixture in FIXTURES {
        let ours = engine.read(&read_data(&format!("{fixture}.png"))).unwrap();
        let theirs = reference_lines(&format!("{fixture}.reference.json"));
        assert_eq!(
            texts(&ours).len(),
            theirs.len(),
            "{fixture}: {:?}",
            texts(&ours)
        );

        for (text, quad, _) in &theirs {
            let line = ours
                .iter()
                .find(|line| &line.text == text)
                .unwrap_or_else(|| panic!("{fixture}: no {text:?} in {:?}", texts(&ours)));
            let error = corner_error(&line.quad, quad);
            assert!(
                error <= 3.0,
                "{fixture} {text:?}: a corner is {error} px off"
            );
        }
    }
}

/// The fixtures include descenders ("Malaysia", "Sundamys", "Layang"), so a
/// box cut too tight shows up as a wrong letter.
#[test]
fn ocr_fixtures_read_in_reading_order() {
    let Some(engine) = load_engine() else {
        eprintln!("{SKIPPED}");
        return;
    };

    for fixture in FIXTURES {
        let ours = engine.read(&read_data(&format!("{fixture}.png"))).unwrap();
        let mut ours = texts(&ours);
        let expected_text =
            String::from_utf8(read_data(&format!("{fixture}.expected.txt"))).unwrap();
        let mut expected: Vec<&str> = expected_text.lines().collect();
        // The margin note's place is ambiguous by design.
        if fixture == "vertical" {
            ours.sort_unstable();
            expected.sort_unstable();
        }
        assert_eq!(ours, expected, "{fixture}");
    }
}

#[test]
fn ocr_label_stored_sideways_with_an_exif_orientation_reads_the_same() {
    let Some(engine) = load_engine() else {
        eprintln!("{SKIPPED}");
        return;
    };

    // A minimal big-endian TIFF block with one entry: orientation (0x0112) = 6,
    // "turn a quarter clockwise to display".
    const ORIENTATION_6: [u8; 26] = [
        b'M', b'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0,
    ];
    let label = read_data("label.png");
    let sideways = image::imageops::rotate270(&image::load_from_memory(&label).unwrap().to_rgb8());
    let mut jpeg = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg, 95);
    encoder.set_exif_metadata(ORIENTATION_6.to_vec()).unwrap();
    let (width, height) = sideways.dimensions();
    encoder
        .write_image(sideways.as_raw(), width, height, ExtendedColorType::Rgb8)
        .unwrap();

    let upright = engine.read(&label).unwrap();
    assert_eq!(texts(&engine.read(&jpeg).unwrap()), texts(&upright));
}

#[test]
fn ocr_non_image_is_an_error_and_a_blank_page_has_no_lines() {
    let Some(engine) = load_engine() else {
        eprintln!("{SKIPPED}");
        return;
    };

    assert!(engine.read(b"Rattus tiomanicus, not an image").is_err());

    let mut blank = Vec::new();
    RgbImage::from_pixel(640, 480, Rgb([255, 255, 255]))
        .write_to(&mut Cursor::new(&mut blank), ImageFormat::Png)
        .unwrap();
    assert_eq!(engine.read(&blank).unwrap(), []);
}

/// Real label photos, compared with PaddleOCR rather than a human
/// transcription: handwriting is ambiguous, and matching PaddleOCR is the goal.
///
/// ocr08 is what catches RGB fed to detection instead of BGR: it loses
/// "Ecological". The coloured label fixture does not, and nothing catches the
/// swap in recognition alone.
#[test]
fn ocr_neon_labels_match_paddleocr() {
    let Some(engine) = load_engine() else {
        eprintln!("{SKIPPED}");
        return;
    };

    if !NEON_PHOTOS.iter().all(|photo| neon_photo(photo).exists()) {
        eprintln!("skipped: the NEON photos are missing — run scripts/fetch-model.sh");
        return;
    }

    let mut failures = Vec::new();
    for photo in NEON_PHOTOS {
        let theirs = reference_lines(&format!("neon/{photo}.reference.json"));
        let ours = engine
            .read(&std::fs::read(neon_photo(photo)).unwrap())
            .unwrap();
        // Both in PaddleOCR's order, so the reading order does not count.
        let kept = |&i: &usize| theirs[i].2 >= MIN_SCORE;
        let our_text: Vec<&str> = in_order_of(&ours, &theirs)
            .into_iter()
            .filter(|(nearest, _)| kept(nearest))
            .map(|(_, text)| text)
            .collect();
        let their_text: Vec<&str> = (0..theirs.len())
            .filter(kept)
            .map(|i| theirs[i].0.as_str())
            .collect();
        let left_out = [theirs.len() - their_text.len(), ours.len() - our_text.len()];
        let (our_text, their_text) = (our_text.join("\n"), their_text.join("\n"));

        let error_rate =
            edit_distance(&our_text, &their_text) as f32 / their_text.chars().count().max(1) as f32;
        // A box cut too tight turns "y" into "v".
        let our_words: HashSet<&str> = our_text.split_whitespace().collect();
        let lost: Vec<&str> = their_text
            .split_whitespace()
            .filter(|word| word.contains(['g', 'j', 'p', 'q', 'y']) && !our_words.contains(word))
            .collect();
        eprintln!(
            "{photo}: character error rate {error_rate:.4}, descender words lost {lost:?}, \
             lines left out (PaddleOCR's, ours) {left_out:?}"
        );

        if error_rate > MAX_ERROR_RATE || !lost.is_empty() {
            failures.push(format!(
                "{photo}: character error rate {error_rate:.4}, lost {lost:?}\nours:\n{our_text}\nPaddleOCR:\n{their_text}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// Our lines in PaddleOCR's order: each goes where the PaddleOCR line nearest
/// to it (by box centre) is, so boxes a few pixels apart cannot reorder them.
/// Each comes with the index of that nearest line.
fn in_order_of<'a>(ours: &'a [TextLine], theirs: &[Reference]) -> Vec<(usize, &'a str)> {
    let centre = |quad: &Quad| [0, 1].map(|axis| quad.iter().map(|p| p[axis]).sum::<f32>() / 4.0);
    let nearest = |quad: &Quad| {
        let [x, y] = centre(quad);
        let distance = |(_, theirs, _): &Reference| {
            let [their_x, their_y] = centre(theirs);
            (x - their_x).hypot(y - their_y)
        };
        (0..theirs.len())
            .min_by(|&a, &b| distance(&theirs[a]).total_cmp(&distance(&theirs[b])))
            .expect("PaddleOCR read some lines")
    };
    let mut sorted: Vec<(usize, &str)> = ours
        .iter()
        .map(|line| (nearest(&line.quad), line.text.as_str()))
        .collect();
    sorted.sort_by_key(|&(nearest, _)| nearest);
    sorted
}

/// `None` only when the models are not compiled in; compiled-in models that
/// fail to load fail the test.
fn load_engine() -> Option<OcrEngine> {
    HAS_OCR.then(|| OcrEngine::load().expect("loading PP-OCRv6"))
}

/// Text, quad and score of a line PaddleOCR read.
type Reference = (String, Quad, f32);

/// PaddleOCR's lines, in its order.
fn reference_lines(file: &str) -> Vec<Reference> {
    let json: serde_json::Value = serde_json::from_slice(&read_data(file)).unwrap();
    let lines = json["lines"].as_array().expect("a lines array");
    let quad = |value: &serde_json::Value| {
        let corner = |i: usize| [0, 1].map(|axis| value[i][axis].as_f64().unwrap() as f32);
        [corner(0), corner(1), corner(2), corner(3)]
    };
    lines
        .iter()
        .map(|line| {
            let score = line["score"].as_f64().unwrap() as f32;
            let text = line["text"].as_str().unwrap().to_string();
            (text, quad(&line["quad"]), score)
        })
        .collect()
}

fn texts(lines: &[TextLine]) -> Vec<&str> {
    lines.iter().map(|line| line.text.as_str()).collect()
}

fn corner_error(ours: &Quad, theirs: &Quad) -> f32 {
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]))
        .fold(0.0, f32::max)
}

/// Levenshtein distance in characters.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    for (i, a_char) in a.chars().enumerate() {
        let mut current = vec![i + 1];
        for (j, &b_char) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(a_char != b_char);
            current.push(substitution.min(previous[j + 1] + 1).min(current[j] + 1));
        }
        previous = current;
    }
    previous[b.len()]
}

fn neon_photo(photo: &str) -> PathBuf {
    Path::new(env!("NAHPU_MODELS_DIR")).join(format!("{photo}.jpg"))
}

fn read_data(file: &str) -> Vec<u8> {
    let path = format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}
