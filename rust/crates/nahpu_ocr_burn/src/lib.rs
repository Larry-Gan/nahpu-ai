//! PP-OCRv6 small on Burn: the text lines in an image, in reading order.
//!
//! Burn runs the detection and recognition models; the steps around them use
//! PaddleOCR 3.7.0's pipeline settings (listed under `params` in each
//! `tests/data/*.reference.json`), not the model ymls' where the two differ.

use anyhow::Result;

#[cfg(has_models)]
mod reading_order;

/// Four corners, clockwise from the top left, in source-image pixels.
pub type Quad = [[f32; 2]; 4];

#[derive(Debug, Clone, PartialEq)]
pub struct TextLine {
    pub text: String,
    /// Mean probability of the recognised characters, 0 to 1.
    pub confidence: f32,
    pub quad: Quad,
}

/// Whether both models are compiled in.
pub const HAS_OCR: bool = cfg!(has_models);

#[cfg(has_models)]
pub use with_models::OcrEngine;

/// Without the models `load` fails, so there is never one to `read` with.
#[cfg(not(has_models))]
pub struct OcrEngine(std::convert::Infallible);

#[cfg(not(has_models))]
impl OcrEngine {
    pub fn load() -> Result<Self> {
        anyhow::bail!("PP-OCRv6 is not compiled in — run scripts/fetch-model.sh PP-OCRv6_small_det PP-OCRv6_small_rec, then rebuild")
    }

    pub fn read(&self, _encoded_image: &[u8]) -> Result<Vec<TextLine>> {
        match self.0 {}
    }
}

#[cfg(has_models)]
mod with_models {
    use super::*;

    use std::path::Path;

    use anyhow::{bail, Context};
    use burn::prelude::*;
    use image::{imageops, GrayImage, Luma, RgbImage};
    use imageproc::contours::find_contours;
    use imageproc::geometric_transformations::{
        warp_into, warp_into_with, Border, Interpolation, Projection,
    };
    use imageproc::geometry::min_area_rect;
    use imageproc::point::Point;
    use nahpu_burn_support::{read_weights, selected_device};
    use nahpu_embed_core::preprocess::decode_upright;

    use crate::reading_order::reading_order;

    #[allow(clippy::all, clippy::pedantic, dead_code)]
    mod det {
        include!(concat!(env!("OUT_DIR"), "/model/PP-OCRv6_small_det.rs"));
    }

    #[allow(clippy::all, clippy::pedantic, dead_code)]
    mod rec {
        include!(concat!(env!("OUT_DIR"), "/model/PP-OCRv6_small_rec.rs"));
    }

    // Detection input. The pipeline scales up only below 64 px and caps the
    // long side at 4000; the mean and std are the det yml's, applied to B, G, R
    // by channel index (PaddleOCR reads images as BGR).
    const DET_MIN_SIDE: f64 = 64.0;
    const DET_MAX_SIDE: f64 = 4000.0;
    const DET_SIDE_MULTIPLE: f64 = 32.0;
    const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

    // Boxes, with the pipeline's 0.3 / 0.6 / 1.5 rather than the det yml's
    // 0.2 / 0.45 / 1.4.
    const TEXT_THRESHOLD: f32 = 0.3;
    const BOX_THRESHOLD: f32 = 0.6;
    const UNCLIP_RATIO: f32 = 1.5;
    const MAX_CANDIDATES: usize = 3000;
    const MIN_BOX_SIDE: f32 = 3.0;

    // Recognition input: height 48, padded to at least 320 wide, squeezed to
    // 3200 when wider.
    const REC_HEIGHT: u32 = 48;
    const REC_MIN_WIDTH: u32 = 320;
    const REC_MAX_WIDTH: u32 = 3200;
    const TALL_CROP_RATIO: f32 = 1.5;

    pub struct OcrEngine {
        det: det::Model,
        rec: rec::Model,
        dictionary: Vec<String>,
        device: Device,
    }

    impl OcrEngine {
        pub fn load() -> Result<Self> {
            let device = selected_device();
            // The recognition model's yml holds its character dictionary.
            let yml = Path::new(env!("NAHPU_MODELS_DIR")).join("PP-OCRv6_small_rec.yml");
            let rec_yml = std::fs::read_to_string(&yml)
                .with_context(|| format!("could not read {}", yml.display()))?;
            let engine = Self {
                det: det::Model::from_bytes(
                    read_weights(env!("PP_OCRV6_SMALL_DET_WEIGHTS"))?,
                    &device,
                ),
                rec: rec::Model::from_bytes(
                    read_weights(env!("PP_OCRV6_SMALL_REC_WEIGHTS"))?,
                    &device,
                ),
                dictionary: dictionary(&rec_yml)?,
                device,
            };

            // Blank first, space last.
            let blank_line = RgbImage::new(REC_MIN_WIDTH, REC_HEIGHT);
            let [_, _, classes] = engine.class_probabilities(&blank_line).dims();
            if engine.dictionary.len() + 2 != classes {
                bail!(
                    "the recognition yml lists {} characters, but the model has {classes} classes (expected characters + 2)",
                    engine.dictionary.len()
                );
            }
            Ok(engine)
        }

        pub fn read(&self, encoded_image: &[u8]) -> Result<Vec<TextLine>> {
            let image = decode_upright(encoded_image)?;

            let mut lines = Vec::new();
            for quad in self.detect(&image) {
                let crop = crop_upright(&image, &quad);
                if crop.width() == 0 || crop.height() == 0 {
                    continue;
                }
                let (text, confidence) = self.recognise(&crop);
                lines.push(Some(TextLine {
                    text,
                    confidence,
                    quad,
                }));
            }

            let quads: Vec<Quad> = lines.iter().flatten().map(|line| line.quad).collect();
            Ok(reading_order(&quads)
                .into_iter()
                .filter_map(|index| lines[index].take())
                .collect())
        }

        fn detect(&self, image: &RgbImage) -> Vec<Quad> {
            let (width, height) = detection_size(image.width(), image.height());
            let resized = resize_linear(image, width, height);
            let pixels = planar_bgr(&resized, width, |byte, channel| {
                (byte as f32 / 255.0 - DET_MEAN[channel]) / DET_STD[channel]
            });
            let input = Tensor::<4>::from_data(
                TensorData::new(pixels, [1, 3, height as usize, width as usize]),
                &self.device,
            );
            let probabilities = to_vec(self.det.forward(input));

            let scale = [
                image.width() as f32 / width as f32,
                image.height() as f32 / height as f32,
            ];
            let limit = [image.width() as f32, image.height() as f32];
            boxes(&probabilities, width, height)
                .into_iter()
                .map(|quad| {
                    quad.map(|point| {
                        [0, 1].map(|axis| {
                            (point[axis] * scale[axis])
                                .round_ties_even()
                                .clamp(0.0, limit[axis])
                        })
                    })
                })
                .collect()
        }

        fn recognise(&self, crop: &RgbImage) -> (String, f32) {
            let probabilities = self.class_probabilities(crop);
            let [_, _, classes] = probabilities.dims();
            ctc_greedy(&to_vec(probabilities), classes, &self.dictionary)
        }

        /// `[1, steps, classes]`, one line at a time: batching pads lines to
        /// the widest and changes the results.
        fn class_probabilities(&self, crop: &RgbImage) -> Tensor<3> {
            let aspect = crop.width() as f64 / crop.height() as f64;
            let padded_width =
                ((REC_HEIGHT as f64 * aspect) as u32).clamp(REC_MIN_WIDTH, REC_MAX_WIDTH);
            let width = ((REC_HEIGHT as f64 * aspect).ceil() as u32).clamp(1, padded_width);

            let resized = resize_linear(crop, width, REC_HEIGHT);
            let pixels = planar_bgr(&resized, padded_width, |byte, _| {
                (byte as f32 / 255.0 - 0.5) / 0.5
            });
            let input = Tensor::<4>::from_data(
                TensorData::new(pixels, [1, 3, REC_HEIGHT as usize, padded_width as usize]),
                &self.device,
            );
            self.rec.forward(input)
        }
    }

    /// PaddleOCR's `DetResizeForTest` as its pipeline runs it.
    fn detection_size(width: u32, height: u32) -> (u32, u32) {
        let (mut width, mut height) = (width as f64, height as f64);
        if width.min(height) < DET_MIN_SIDE {
            let ratio = DET_MIN_SIDE / width.min(height);
            (width, height) = ((width * ratio).trunc(), (height * ratio).trunc());
        }
        if width.max(height) > DET_MAX_SIDE {
            let ratio = DET_MAX_SIDE / width.max(height);
            (width, height) = ((width * ratio).trunc(), (height * ratio).trunc());
        }
        // Python's round, half to even.
        let round = |side: f64| {
            ((side / DET_SIDE_MULTIPLE).round_ties_even() * DET_SIDE_MULTIPLE)
                .max(DET_SIDE_MULTIPLE)
        };
        (round(width) as u32, round(height) as u32)
    }

    /// Bilinear with pixel centres at half-pixel offsets, as OpenCV's
    /// `INTER_LINEAR`, which PaddleOCR uses. Unlike the `image` crate's
    /// Triangle filter it does not smooth when shrinking; that smoothing moved
    /// boxes and changed letters on the NEON photos.
    fn resize_linear(image: &RgbImage, width: u32, height: u32) -> RgbImage {
        let scale_x = image.width() as f32 / width as f32;
        let scale_y = image.height() as f32 / height as f32;
        let source = move |x: f32, y: f32| ((x + 0.5) * scale_x - 0.5, (y + 0.5) * scale_y - 0.5);
        let mut resized = RgbImage::new(width, height);
        warp_into_with(
            image,
            source,
            Interpolation::Bilinear,
            Border::Replicate,
            &mut resized,
        );
        resized
    }

    /// Planar `3 × height × width` in BGR order; columns right of the image
    /// stay 0.0, PaddleOCR's padding after normalising.
    fn planar_bgr(image: &RgbImage, width: u32, normalise: impl Fn(u8, usize) -> f32) -> Vec<f32> {
        let plane = (width * image.height()) as usize;
        let mut values = vec![0.0; 3 * plane];
        for (x, y, pixel) in image.enumerate_pixels() {
            for bgr in 0..3 {
                values[bgr * plane + (y * width + x) as usize] = normalise(pixel[2 - bgr], bgr);
            }
        }
        values
    }

    /// Boxes in probability-map pixels. Each region above the text threshold
    /// gets its minimum-area rectangle, which is scored by the mean
    /// probability inside it and grown by `area × ratio / perimeter` on every
    /// side (PaddleOCR's unclip, which for a rectangle is a larger rectangle).
    fn boxes(probabilities: &[f32], width: u32, height: u32) -> Vec<Quad> {
        let text = GrayImage::from_fn(width, height, |x, y| {
            Luma([u8::from(
                probabilities[(y * width + x) as usize] > TEXT_THRESHOLD,
            )])
        });

        let mut quads = Vec::new();
        for contour in find_contours::<i32>(&text).into_iter().take(MAX_CANDIDATES) {
            let rect = Rect::around(&contour.points);
            let [long, short] = rect.half.map(|half| 2.0 * half);
            if long.min(short) < MIN_BOX_SIDE
                || rect.score(probabilities, width, height) < BOX_THRESHOLD
            {
                continue;
            }
            // Evenly on every side: capping the vertical growth cuts off
            // descenders, so "y" reads as "v".
            quads.push(rect.corners(long * short * UNCLIP_RATIO / (2.0 * (long + short))));
        }
        quads
    }

    /// A rotated rectangle: its centre, the unit vector along its top edge,
    /// and half its width and height.
    struct Rect {
        centre: [f32; 2],
        across: [f32; 2],
        half: [f32; 2],
    }

    impl Rect {
        /// imageproc rounds the corners of its minimum-area rectangle outward
        /// to whole pixels, which skews a tilted one; so only its angle is
        /// used, and the extent is measured from the points.
        fn around(points: &[Point<i32>]) -> Self {
            let [top_left, top_right, ..] = min_area_rect(points);
            let (dx, dy) = (
                (top_right.x - top_left.x) as f32,
                (top_right.y - top_left.y) as f32,
            );
            let length = dx.hypot(dy);
            let across = if length > 0.0 {
                [dx / length, dy / length]
            } else {
                [1.0, 0.0]
            };
            let down = [-across[1], across[0]];
            let extent = |axis: [f32; 2]| {
                let (low, high) =
                    min_max(points.iter().map(|p| dot([p.x as f32, p.y as f32], axis)));
                ((low + high) / 2.0, (high - low) / 2.0)
            };
            let ((along, half_width), (below, half_height)) = (extent(across), extent(down));
            Self {
                centre: [0, 1].map(|i| along * across[i] + below * down[i]),
                across,
                half: [half_width, half_height],
            }
        }

        /// Corners clockwise from the top left, every side moved out by `by`.
        fn corners(&self, by: f32) -> Quad {
            let down = [-self.across[1], self.across[0]];
            let [width, height] = self.half.map(|half| half + by);
            [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(x, y)| {
                [0, 1].map(|i| self.centre[i] + x * width * self.across[i] + y * height * down[i])
            })
        }

        /// Mean probability of the map pixels inside, edges included.
        fn score(&self, probabilities: &[f32], width: u32, height: u32) -> f32 {
            let corners = self.corners(0.0);
            let span = |axis: usize, limit: u32| {
                let (low, high) = min_max(corners.iter().map(|corner| corner[axis]));
                (low.floor().max(0.0) as u32)..=(high.ceil().min(limit as f32 - 1.0) as u32)
            };
            let down = [-self.across[1], self.across[0]];

            let (mut sum, mut count) = (0.0, 0);
            for y in span(1, height) {
                for x in span(0, width) {
                    let offset = [x as f32 - self.centre[0], y as f32 - self.centre[1]];
                    if dot(offset, self.across).abs() <= self.half[0] + 1e-3
                        && dot(offset, down).abs() <= self.half[1] + 1e-3
                    {
                        sum += probabilities[(y * width + x) as usize];
                        count += 1;
                    }
                }
            }
            if count == 0 {
                0.0
            } else {
                sum / count as f32
            }
        }
    }

    fn min_max(values: impl Iterator<Item = f32>) -> (f32, f32) {
        values.fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), v| {
            (low.min(v), high.max(v))
        })
    }

    fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
        a[0] * b[0] + a[1] * b[1]
    }

    /// Cuts the quad out upright and turns a tall crop (vertical text) a
    /// quarter counter-clockwise, as PaddleOCR does.
    fn crop_upright(image: &RgbImage, quad: &Quad) -> RgbImage {
        let [top_left, top_right, bottom_right, bottom_left] = *quad;
        let width = distance(top_left, top_right).max(distance(bottom_left, bottom_right)) as u32;
        let height = distance(top_left, bottom_left).max(distance(top_right, bottom_right)) as u32;

        let axis_aligned = top_left[1] == top_right[1]
            && bottom_left[1] == bottom_right[1]
            && top_left[0] == bottom_left[0]
            && top_right[0] == bottom_right[0];
        let crop = if axis_aligned {
            let [x, y] = top_left.map(|v| v as u32);
            imageops::crop_imm(image, x, y, width, height).to_image()
        } else {
            let (w, h) = (width as f32, height as f32);
            let corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
            let mut crop = RgbImage::new(width, height);
            if let Some(projection) =
                Projection::from_control_points(quad.map(|[x, y]| (x, y)), corners)
            {
                warp_into(
                    image,
                    projection,
                    Interpolation::Bicubic,
                    Border::Replicate,
                    &mut crop,
                );
            }
            crop
        };

        if crop.height() as f32 >= TALL_CROP_RATIO * crop.width() as f32 {
            imageops::rotate270(&crop)
        } else {
            crop
        }
    }

    /// Greedy CTC: the likeliest class at each step, repeats merged, the blank
    /// (class 0) dropped. Class `i` is `dictionary[i - 1]`, and the last class
    /// is a space. The confidence is the mean probability of the kept steps.
    fn ctc_greedy(probabilities: &[f32], classes: usize, dictionary: &[String]) -> (String, f32) {
        let (mut text, mut kept, mut previous) = (String::new(), Vec::new(), 0);
        for step in probabilities.chunks_exact(classes) {
            // The first of equal maxima, as numpy's argmax.
            let best =
                |best: (usize, f32), next: (usize, f32)| if next.1 > best.1 { next } else { best };
            let (class, probability) = step.iter().copied().enumerate().fold((0, f32::MIN), best);
            if class != 0 && class != previous {
                text.push_str(dictionary.get(class - 1).map_or(" ", String::as_str));
                kept.push(probability);
            }
            previous = class;
        }
        let confidence = if kept.is_empty() {
            0.0
        } else {
            kept.iter().sum::<f32>() / kept.len() as f32
        };
        (text, confidence)
    }

    /// The list under `PostProcess:` → `character_dict:`, one `  - ` entry per
    /// line, some single-quoted (`''` is a quote).
    fn dictionary(rec_yml: &str) -> Result<Vec<String>> {
        let entries: Vec<String> = rec_yml
            .lines()
            .skip_while(|line| *line != "PostProcess:")
            .skip_while(|line| *line != "  character_dict:")
            .skip(1)
            .map_while(|line| line.strip_prefix("  - "))
            .map(
                |entry| match entry.strip_prefix('\'').and_then(|e| e.strip_suffix('\'')) {
                    Some(quoted) => quoted.replace("''", "'"),
                    None => entry.to_string(),
                },
            )
            .collect();
        if entries.is_empty() {
            bail!("no PostProcess character_dict in the recognition yml");
        }
        Ok(entries)
    }

    fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
        (a[0] - b[0]).hypot(a[1] - b[1])
    }

    fn to_vec<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .expect("model output is f32")
    }
}
