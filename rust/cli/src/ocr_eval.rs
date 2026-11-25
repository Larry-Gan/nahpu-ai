//! Reads label photos and compares the text with what PaddleOCR read on the
//! same photos, then writes metrics and a Markdown report.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use nahpu_embed_burn::BACKEND_NAME;
use nahpu_ocr_burn::{OcrEngine, Quad, TextLine};

/// PaddleOCR's lines scored below this are left out, with ours nearest to
/// them, as in the OCR crate's own test. On these photos they are ruler tick
/// marks and a sideways logo.
const MIN_REFERENCE_SCORE: f32 = 0.5;

pub struct OcrEvalPlan {
    pub references: PathBuf,
    pub photos: PathBuf,
    pub out: PathBuf,
}

/// A `*.reference.json` written by `scripts/ocr/make_references.py`.
#[derive(Deserialize)]
struct Reference {
    fixture: String,
    image_width: u32,
    image_height: u32,
    packages: Packages,
    lines: Vec<ReferenceLine>,
}

#[derive(Deserialize)]
struct Packages {
    paddleocr: String,
}

#[derive(Deserialize)]
struct ReferenceLine {
    text: String,
    score: f32,
    quad: Quad,
}

#[derive(Serialize)]
struct Summary {
    model: &'static str,
    backend: &'static str,
    compared_with: String,
    references: String,
    photos: usize,
    load_seconds: f64,
    photo_ms_median: f64,
    min_reference_score: f32,
    runs: Vec<PhotoMetrics>,
}

#[derive(Serialize)]
struct PhotoMetrics {
    photo: String,
    width: u32,
    height: u32,
    lines: usize,
    reference_lines: usize,
    character_error_rate: f64,
    lines_left_out: usize,
    reference_lines_left_out: usize,
    milliseconds: f64,
    /// What was read, in reading order.
    text: Vec<String>,
    /// The compared lines that differ, in PaddleOCR's order: its text, then ours.
    differences: Vec<[String; 2]>,
}

pub fn run(plan: &OcrEvalPlan) -> Result<()> {
    let references = read_references(plan)?;

    let started = Instant::now();
    let engine = OcrEngine::load()?;
    let load_seconds = started.elapsed().as_secs_f64();

    let mut runs = Vec::new();
    for reference in &references {
        let photo = plan.photos.join(&reference.fixture);
        let encoded = std::fs::read(&photo)
            .with_context(|| format!("could not read {} — run scripts/fetch-model.sh", photo.display()))?;
        let started = Instant::now();
        let lines = engine.read(&encoded).with_context(|| format!("could not read text in {}", photo.display()))?;
        let milliseconds = started.elapsed().as_secs_f64() * 1000.0;
        runs.push(compare(reference, &lines, milliseconds));
        eprintln!("  {}", reference.fixture);
    }

    let mut timings: Vec<f64> = runs.iter().map(|run| run.milliseconds).collect();
    timings.sort_by(f64::total_cmp);
    let summary = Summary {
        model: "PP-OCRv6 small",
        backend: BACKEND_NAME,
        compared_with: format!("PaddleOCR {}", references[0].packages.paddleocr),
        references: plan.references.display().to_string(),
        photos: runs.len(),
        load_seconds,
        photo_ms_median: timings[(timings.len() - 1) / 2],
        min_reference_score: MIN_REFERENCE_SCORE,
        runs,
    };
    std::fs::create_dir_all(&plan.out)?;
    std::fs::write(plan.out.join("metrics.json"), serde_json::to_string_pretty(&summary)?)?;
    std::fs::write(plan.out.join("report.md"), report(&summary))?;
    eprintln!("wrote {}", plan.out.display());
    Ok(())
}

/// Every `*.reference.json` in the folder, by file name.
fn read_references(plan: &OcrEvalPlan) -> Result<Vec<Reference>> {
    let folder = &plan.references;
    let mut files: Vec<PathBuf> = std::fs::read_dir(folder)
        .with_context(|| format!("could not read {}", folder.display()))?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<_>>()?;
    files.retain(|file| file.to_string_lossy().ends_with(".reference.json"));
    files.sort();
    if files.is_empty() {
        bail!("{} has no *.reference.json", folder.display());
    }

    files
        .iter()
        .map(|file| {
            let reference: Reference = serde_json::from_slice(&std::fs::read(file)?)
                .with_context(|| format!("could not parse {}", file.display()))?;
            if reference.lines.is_empty() {
                bail!("{} lists no lines", file.display());
            }
            Ok(reference)
        })
        .collect()
}

/// The comparison `tests/ocr.rs` gates on: both texts in PaddleOCR's order, so
/// the reading order does not count.
fn compare(reference: &Reference, ours: &[TextLine], milliseconds: f64) -> PhotoMetrics {
    let theirs = &reference.lines;
    let kept = |i: usize| theirs[i].score >= MIN_REFERENCE_SCORE;

    // Each of our lines goes where the PaddleOCR line nearest to it (by box
    // centre) is, so boxes a few pixels apart cannot reorder them.
    let mut placed: Vec<(usize, &str)> =
        ours.iter().map(|line| (nearest(&line.quad, theirs), line.text.as_str())).collect();
    placed.sort_by_key(|&(index, _)| index);
    let our_lines: Vec<&str> = placed.iter().filter(|&&(index, _)| kept(index)).map(|&(_, text)| text).collect();
    let their_lines: Vec<&str> = (0..theirs.len()).filter(|&i| kept(i)).map(|i| theirs[i].text.as_str()).collect();

    let differences = (0..theirs.len())
        .filter(|&i| kept(i))
        .filter_map(|i| {
            let here: Vec<&str> = placed.iter().filter(|&&(index, _)| index == i).map(|&(_, text)| text).collect();
            let here = here.join(" ");
            (here != theirs[i].text).then(|| [theirs[i].text.clone(), here])
        })
        .collect();

    let (our_text, their_text) = (our_lines.join("\n"), their_lines.join("\n"));
    PhotoMetrics {
        photo: reference.fixture.clone(),
        width: reference.image_width,
        height: reference.image_height,
        lines: ours.len(),
        reference_lines: theirs.len(),
        character_error_rate: edit_distance(&our_text, &their_text) as f64 / their_text.chars().count().max(1) as f64,
        lines_left_out: ours.len() - our_lines.len(),
        reference_lines_left_out: theirs.len() - their_lines.len(),
        milliseconds,
        // A detected box in which nothing was recognised has no text to show.
        text: ours.iter().map(|line| line.text.clone()).filter(|text| !text.trim().is_empty()).collect(),
        differences,
    }
}

fn nearest(quad: &Quad, theirs: &[ReferenceLine]) -> usize {
    let centre = |quad: &Quad| [0, 1].map(|axis| quad.iter().map(|corner| corner[axis]).sum::<f32>() / 4.0);
    let [x, y] = centre(quad);
    let distance = |line: &ReferenceLine| {
        let [their_x, their_y] = centre(&line.quad);
        (x - their_x).hypot(y - their_y)
    };
    (0..theirs.len())
        .min_by(|&a, &b| distance(&theirs[a]).total_cmp(&distance(&theirs[b])))
        .expect("references with no lines are refused when read")
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

fn report(summary: &Summary) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# OCR evaluation: {}\n", summary.model);
    let _ = writeln!(md, "- Backend: {}", summary.backend);
    let _ = writeln!(
        md,
        "- Compared with: {} on the same ONNX files, from `{}`, {} photos",
        summary.compared_with, summary.references, summary.photos
    );
    let _ = writeln!(
        md,
        "- Model load {:.2} s; per photo (decode + detect + recognise) median {:.0} ms\n",
        summary.load_seconds, summary.photo_ms_median
    );
    let _ = writeln!(
        md,
        "The character error rate is the edit distance to PaddleOCR's text over its length, with our lines put in \
         PaddleOCR's order. It says how closely this engine follows PaddleOCR, not how well either reads the label: \
         there is no human transcription. PaddleOCR's lines scored below {} are left out, with ours nearest to them.\n",
        summary.min_reference_score
    );

    let _ = writeln!(md, "| Photo | Size | Lines | PaddleOCR's lines | Character error rate | Left out (ours, PaddleOCR's) | Time |");
    let _ = writeln!(md, "|---|---|---|---|---|---|---|");
    for run in &summary.runs {
        let _ = writeln!(
            md,
            "| {} | {} x {} | {} | {} | {:.1}% | {}, {} | {:.0} ms |",
            run.photo,
            run.width,
            run.height,
            run.lines,
            run.reference_lines,
            run.character_error_rate * 100.0,
            run.lines_left_out,
            run.reference_lines_left_out,
            run.milliseconds
        );
    }

    for run in &summary.runs {
        let _ = writeln!(md, "\n## {}\n", run.photo);
        let _ = writeln!(md, "Read, in reading order:\n\n```\n{}\n```\n", run.text.join("\n"));
        if run.differences.is_empty() {
            let _ = writeln!(md, "Every compared line matches PaddleOCR.");
            continue;
        }
        let _ = writeln!(md, "Compared lines that differ:\n");
        let _ = writeln!(md, "| PaddleOCR | Ours |");
        let _ = writeln!(md, "|---|---|");
        for [theirs, ours] in &run.differences {
            let _ = writeln!(md, "| {} | {} |", table_cell(theirs), table_cell(ours));
        }
    }
    md
}

fn table_cell(text: &str) -> String {
    if text.is_empty() {
        "(nothing)".into()
    } else {
        text.replace('|', "\\|")
    }
}
