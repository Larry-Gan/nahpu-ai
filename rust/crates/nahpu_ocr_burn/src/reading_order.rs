//! Reading order for the detected lines: a recursive XY-cut over their
//! bounding rectangles. PaddleOCR has none; its own order interleaves columns.

use crate::Quad;

const GUTTER_LINE_HEIGHTS: f32 = 1.5;
const MIN_COLUMN_LINES: usize = 3;
const MIN_COLUMN_WIDTH_LINE_HEIGHTS: f32 = 8.0;
const MIN_SECTION_GAP_LINE_HEIGHTS: f32 = 0.5;
const SECTION_GAP_OVER_LINE_GAP: f32 = 1.5;
const SAME_ROW_MIN_OVERLAP: f32 = 0.5;

type Split = (Vec<usize>, Vec<usize>);

/// Indices into `quads`, in the order a person would read them.
///
/// A recursive XY-cut: split at a column gutter if there is one, otherwise at an
/// unusually wide gap between lines, otherwise read row by row. Coordinates are
/// assumed finite.
pub fn reading_order(quads: &[Quad]) -> Vec<usize> {
    Page::of(quads).xy_cut((0..quads.len()).collect())
}

#[derive(Debug, Clone, Copy)]
struct Bounds {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

impl Bounds {
    fn of(quad: &Quad) -> Self {
        let xs = quad.map(|[x, _]| x);
        let ys = quad.map(|[_, y]| y);
        Self {
            left: xs.into_iter().fold(f32::INFINITY, f32::min),
            top: ys.into_iter().fold(f32::INFINITY, f32::min),
            right: xs.into_iter().fold(f32::NEG_INFINITY, f32::max),
            bottom: ys.into_iter().fold(f32::NEG_INFINITY, f32::max),
        }
    }
}

struct Page {
    bounds: Vec<Bounds>,
    line_height: f32,
    typical_line_gap: f32,
}

impl Page {
    fn of(quads: &[Quad]) -> Self {
        let bounds: Vec<Bounds> = quads.iter().map(Bounds::of).collect();
        let line_height = median(bounds.iter().map(|b| b.bottom - b.top).collect());
        let typical_line_gap = median(
            bounds
                .iter()
                .filter_map(|above| gap_to_line_below(above, &bounds))
                .collect(),
        );
        Self {
            bounds,
            line_height,
            typical_line_gap,
        }
    }

    fn xy_cut(&self, region: Vec<usize>) -> Vec<usize> {
        match self
            .column_split(&region)
            .or_else(|| self.section_split(&region))
        {
            Some((first, second)) => [self.xy_cut(first), self.xy_cut(second)].concat(),
            None => self.rows(&region).concat(),
        }
    }

    fn column_split(&self, region: &[usize]) -> Option<Split> {
        let (by_left, gaps) = self.gaps_along(region, |b| b.left, |b| b.right);
        gaps.into_iter()
            .filter(|&(gap, at)| {
                gap >= GUTTER_LINE_HEIGHTS * self.line_height
                    && self.is_column(&by_left[..at])
                    && self.is_column(&by_left[at..])
            })
            .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)))
            .map(|(_, at)| split(by_left, at))
    }

    fn section_split(&self, region: &[usize]) -> Option<Split> {
        let (by_top, gaps) = self.gaps_along(region, |b| b.top, |b| b.bottom);
        let section_gap = (MIN_SECTION_GAP_LINE_HEIGHTS * self.line_height)
            .max(SECTION_GAP_OVER_LINE_GAP * self.typical_line_gap);
        let (_, at) = gaps.into_iter().find(|&(gap, _)| gap >= section_gap)?;
        Some(split(by_top, at))
    }

    fn is_column(&self, side: &[usize]) -> bool {
        let rows = self.rows(side);
        let widths = rows.iter().map(|row| self.row_width(row)).collect();
        rows.len() >= MIN_COLUMN_LINES
            && median(widths) >= MIN_COLUMN_WIDTH_LINE_HEIGHTS * self.line_height
    }

    /// Region sorted by `start`, and every positive gap as (width, index of the first box after it).
    fn gaps_along(
        &self,
        region: &[usize],
        start: fn(&Bounds) -> f32,
        end: fn(&Bounds) -> f32,
    ) -> (Vec<usize>, Vec<(f32, usize)>) {
        let sorted = self.sorted_by(region, start);
        let mut reach = f32::NEG_INFINITY;
        let gaps = (1..sorted.len())
            .filter_map(|at| {
                reach = reach.max(end(&self.bounds[sorted[at - 1]]));
                let gap = start(&self.bounds[sorted[at]]) - reach;
                (gap > 0.0).then_some((gap, at))
            })
            .collect();
        (sorted, gaps)
    }

    /// Boxes share a row when they overlap the band common to the row by at least
    /// half the shorter height; a tall box cannot glue two lines together.
    fn rows(&self, region: &[usize]) -> Vec<Vec<usize>> {
        let mut rows: Vec<Vec<usize>> = Vec::new();
        let mut band = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for index in self.sorted_by(region, |b| b.top) {
            let b = self.bounds[index];
            match rows.last_mut() {
                Some(row) if shares_band(band, &b) => {
                    row.push(index);
                    band = (band.0.max(b.top), band.1.min(b.bottom));
                }
                _ => {
                    rows.push(vec![index]);
                    band = (b.top, b.bottom);
                }
            }
        }
        for row in &mut rows {
            *row = self.sorted_by(row, |b| b.left);
        }
        rows
    }

    fn row_width(&self, row: &[usize]) -> f32 {
        let left = row
            .iter()
            .map(|&i| self.bounds[i].left)
            .fold(f32::INFINITY, f32::min);
        let right = row
            .iter()
            .map(|&i| self.bounds[i].right)
            .fold(f32::NEG_INFINITY, f32::max);
        right - left
    }

    fn sorted_by(&self, region: &[usize], key: fn(&Bounds) -> f32) -> Vec<usize> {
        let mut sorted = region.to_vec();
        sorted.sort_by(|&a, &b| {
            key(&self.bounds[a])
                .total_cmp(&key(&self.bounds[b]))
                .then(a.cmp(&b))
        });
        sorted
    }
}

fn shares_band((top, bottom): (f32, f32), b: &Bounds) -> bool {
    let overlap = bottom.min(b.bottom) - top.max(b.top);
    let shorter = (bottom - top).min(b.bottom - b.top);
    overlap >= SAME_ROW_MIN_OVERLAP * shorter
}

/// Distance from `above` to the nearest box starting below it that shares some of its width.
fn gap_to_line_below(above: &Bounds, bounds: &[Bounds]) -> Option<f32> {
    bounds
        .iter()
        .filter(|b| b.top >= above.bottom && b.left < above.right && above.left < b.right)
        .map(|b| b.top - above.bottom)
        .min_by(f32::total_cmp)
}

fn split(mut indices: Vec<usize>, at: usize) -> Split {
    let second = indices.split_off(at);
    (indices, second)
}

fn median(mut values: Vec<f32>) -> f32 {
    values.sort_by(f32::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}
