mod common;

use std::io::Cursor;

use common::{fixture_photos_present, read_fixture_photo, reference_rows};
use image::ImageFormat;
use nahpu_embed_burn::{load_nomic_text, load_nomic_vision, HAS_NOMIC};
use nahpu_embed_core::{cosine_similarity, TextRole};

const TEXT_REFERENCE: &str = include_str!("data/nomic-text-reference.tsv");
const TENSOR_REFERENCE: &str = include_str!("data/nomic-vision-tensor-reference.tsv");
const IMAGE_REFERENCE: &str = include_str!("data/nomic-vision-image-reference.tsv");
/// The NEON photos, each with the query that describes it. They are fetched
/// by `scripts/fetch-model.sh`, like the other fixture photos.
const PHOTOS: [(&str, &str); 6] = [
    (
        "emb01_beetle_chlaenius_aestivus_dorsal",
        "a photo of a beetle",
    ),
    (
        "emb05_mosquito_aedes_monticola_lateral",
        "a photo of a mosquito",
    ),
    (
        "emb06_lizard_holbrookia_maculata_dorsal",
        "a photo of a lizard",
    ),
    (
        "emb07_salamander_plethodon_cinereus_three",
        "a photo of a salamander",
    ),
    ("emb08_toad_anaxyrus_americanus_two", "a photo of a toad"),
    ("emb10_mammal_peromyscus_ventral", "a photo of a mouse"),
];
/// Left out of the matching assertion (its row is still printed), because
/// Nomic itself gets it wrong: it ranks this lizard (small, dark, on the same
/// foam-and-ruler setup as the salamanders and toads) as a salamander, its own
/// query only third (salamander 0.0759, toad 0.0646, lizard 0.0484). It still
/// checks against its reference and carries the EXIF test.
const LIZARD: &str = "emb06_lizard_holbrookia_maculata_dorsal";

#[test]
fn nomic_text_matches_the_onnx_reference() {
    if skip_without_nomic() {
        return;
    }
    let embedder = load_nomic_text().unwrap();

    for fields in rows(TEXT_REFERENCE) {
        let (role, sentence, expected) = (fields[0], fields[1], parse_vector(fields[2]));
        let text_role = match role {
            "query" => TextRole::Query,
            "document" => TextRole::Document,
            other => panic!("unknown role {other:?}"),
        };
        let actual = embedder.embed(sentence, text_role).unwrap();

        // The cosine alone would miss a forgotten layer norm.
        let cosine = cosine_similarity(&actual, &expected);
        let max_difference = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            cosine > 0.999 && max_difference <= 1e-4,
            "{role} {sentence:?}: cosine {cosine}, max difference {max_difference}"
        );
    }
}

/// Feeds the model a known tensor, so an import fault shows apart from any
/// difference in preprocessing.
#[test]
fn nomic_vision_matches_the_onnx_reference_on_a_synthetic_tensor() {
    if skip_without_nomic() {
        return;
    }
    let embedder = load_nomic_vision().unwrap();
    let config = embedder.preprocess_config();
    let side = config.size as usize;

    let index = |c: usize, y: usize, x: usize| (c * side + y) * side + x;
    let mut pixels = vec![0.0; 3 * side * side];
    for c in 0..3 {
        for y in 0..side {
            for x in 0..side {
                let byte = ((x * 7 + y * 13 + c * 29) % 256) as u8;
                pixels[index(c, y, x)] = (f32::from(byte) / 255.0 - config.mean[c]) / config.std[c];
            }
        }
    }

    let mut expected = Vec::new();
    for fields in rows(TENSOR_REFERENCE) {
        match fields[0] {
            "input" => {
                let at: Vec<usize> = fields[1..4].iter().map(|f| f.parse().unwrap()).collect();
                let value: f32 = fields[4].parse().unwrap();
                assert_eq!(pixels[index(at[0], at[1], at[2])], value, "input at {at:?}");
            }
            "embedding" => expected = parse_vector(fields[1]),
            _ => {}
        }
    }

    let actual = embedder.embed_pixels(&pixels).unwrap();
    let cosine = cosine_similarity(&actual, &expected);
    assert!(cosine > 0.999, "cosine {cosine}");
}

/// The references were resized with Pillow, which differs slightly from the
/// `image` crate (real scores are at least 0.9990).
#[test]
fn nomic_photos_match_the_onnx_references() {
    if skip_without_nomic() || skip_without_photos() {
        return;
    }
    let embedder = load_nomic_vision().unwrap();

    let references = reference_rows(IMAGE_REFERENCE);
    assert_eq!(references.len(), 6, "expected six reference photos");

    for (file, expected) in references {
        let actual = embedder.embed_image(&read_fixture_photo(&file)).unwrap();

        let cosine = cosine_similarity(&actual, &expected);
        eprintln!("{file}: cosine {cosine}");
        assert!(cosine > 0.995, "{file}: cosine {cosine}");
    }
}

/// If the orientation tag were ignored, the tagged photo would match the
/// unrotated photo instead.
#[test]
fn nomic_photo_with_an_exif_orientation_matches_it_rotated() {
    if skip_without_nomic() || skip_without_photos() {
        return;
    }
    let embedder = load_nomic_vision().unwrap();

    // A minimal big-endian TIFF block with one entry: orientation (0x0112) = 6,
    // "turn a quarter clockwise to display".
    const ORIENTATION_6: [u8; 26] = [
        b'M', b'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0,
    ];
    let photo = read_fixture_photo(LIZARD);
    assert!(
        !photo.windows(6).any(|bytes| bytes == b"Exif\0\0"),
        "the photo already has EXIF data"
    );
    // An APP1 segment (marker, length, "Exif\0\0", TIFF block) right after the
    // start-of-image marker.
    let length = (2 + 6 + ORIENTATION_6.len()) as u16;
    let mut tagged = photo[..2].to_vec();
    tagged.extend([0xFF, 0xE1]);
    tagged.extend(length.to_be_bytes());
    tagged.extend(b"Exif\0\0");
    tagged.extend(ORIENTATION_6);
    tagged.extend(&photo[2..]);

    let decoded = image::load_from_memory(&photo).unwrap().to_rgb8();
    let mut rotated = Vec::new();
    image::imageops::rotate90(&decoded)
        .write_to(&mut Cursor::new(&mut rotated), ImageFormat::Png)
        .unwrap();

    let embed = |bytes: &[u8]| embedder.embed_image(bytes).unwrap();
    let tagged = embed(&tagged);
    let cosine = cosine_similarity(&tagged, &embed(&rotated));
    let unrotated = cosine_similarity(&tagged, &embed(&photo));
    eprintln!("cosine {cosine} with the photo rotated in code, {unrotated} with it unrotated");
    assert!(cosine > 0.9999, "cosine {cosine}");
}

#[test]
fn nomic_each_photo_matches_its_own_query() {
    if skip_without_nomic() || skip_without_photos() {
        return;
    }
    let text_embedder = load_nomic_text().unwrap();
    let image_embedder = load_nomic_vision().unwrap();

    let queries: Vec<_> = PHOTOS
        .iter()
        .map(|(_, query)| text_embedder.embed(query, TextRole::Query).unwrap())
        .collect();

    // One row per photo, one column per query.
    let columns: String = PHOTOS
        .iter()
        .map(|(_, query)| format!("{:>11}", query.rsplit(' ').next().unwrap()))
        .collect();
    eprintln!("{:46}{columns}", "");
    let mut mismatches = Vec::new();
    for (photo, query) in PHOTOS {
        let image = image_embedder
            .embed_image(&read_fixture_photo(photo))
            .unwrap();
        let scores: Vec<f32> = queries
            .iter()
            .map(|text| cosine_similarity(&image, text))
            .collect();
        let row: String = scores
            .iter()
            .map(|score| format!("{score:>11.4}"))
            .collect();
        eprintln!("{photo:46}{row}");

        let best = (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        if photo != LIZARD && PHOTOS[best].1 != query {
            mismatches.push(format!("{photo}: best query {:?}", PHOTOS[best].1));
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// Skips only when a model is not compiled in; a compiled-in model that fails
/// to load fails the test.
fn skip_without_nomic() -> bool {
    if HAS_NOMIC {
        return false;
    }
    eprintln!("skipped: the Nomic models are not compiled in — run scripts/fetch-model.sh nomic-embed-text-v1_5 nomic-embed-vision-v1_5, then rebuild");
    true
}

fn skip_without_photos() -> bool {
    if fixture_photos_present(IMAGE_REFERENCE) {
        return false;
    }
    eprintln!("skipped: the NEON photos are missing — run scripts/fetch-model.sh");
    true
}

fn rows(table: &str) -> Vec<Vec<&str>> {
    table
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| line.split('\t').collect())
        .collect()
}

fn parse_vector(text: &str) -> Vec<f32> {
    text.split(',').map(|x| x.parse().expect("float")).collect()
}
