//! Explicit training-only separator search or quadratic moment initialization.
//! These are declared biases, not CE gradients or unseen-label certificates.
use crate::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Direction {
    pub axis: [f64; 2],
    pub lower: usize,
    pub upper: usize,
    pub gap: f64,
    pub threshold: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockReport {
    pub block: usize,
    pub labels: Vec<usize>,
    pub records: usize,
    pub method: String,
    pub candidates: Vec<Direction>,
    pub selected: Option<usize>,
    pub margin_scale: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub blocks: Vec<BlockReport>,
    pub searched_directions: usize,
    pub direction_records: usize,
    pub moment_records: usize,
    pub materialized_coefficients: usize,
    pub projected_coefficients: usize,
}
pub fn directions() -> Result<Vec<[f64; 2]>> {
    let p: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    let raw: Vec<[f64; 2]> = serde_json::from_value(p["directions"].clone())?;
    Ok(raw
        .into_iter()
        .map(|[x, y]| {
            let n = x.hypot(y);
            [x / n, y / n]
        })
        .collect())
}
fn candidate(n: usize, groups: &BTreeMap<usize, Vec<&Record>>, axis: [f64; 2]) -> Direction {
    let intervals: Vec<_> = groups
        .iter()
        .map(|(&label, rows)| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for r in rows {
                let [u, v] = features::point(n, r.cell);
                let d = axis[0] * u + axis[1] * v;
                lo = lo.min(d);
                hi = hi.max(d);
            }
            (label, lo, hi)
        })
        .collect();
    let (a, b) = (intervals[0], intervals[1]);
    if b.1 - a.2 >= a.1 - b.2 {
        Direction {
            axis,
            lower: a.0,
            upper: b.0,
            gap: b.1 - a.2,
            threshold: (b.1 + a.2) / 2.,
        }
    } else {
        Direction {
            axis,
            lower: b.0,
            upper: a.0,
            gap: a.1 - b.2,
            threshold: (a.1 + b.2) / 2.,
        }
    }
}
fn moments(n: usize, rows: &[&Record], total: usize) -> Result<[f64; STATES]> {
    let count = rows.len() as f64;
    let mut mean = [0.; 2];
    for r in rows {
        let p = features::point(n, r.cell);
        for i in 0..2 {
            mean[i] += p[i] / count;
        }
    }
    let (mut a, mut b, mut c) = (0.01, 0., 0.01);
    for r in rows {
        let p = features::point(n, r.cell);
        let (u, v) = (p[0] - mean[0], p[1] - mean[1]);
        a += u * u / count;
        b += u * v / count;
        c += v * v / count;
    }
    let det = a * c - b * b;
    if !det.is_finite() || det <= 0. {
        return Err("quadratic moment covariance refused".into());
    }
    let (aa, bb, cc) = (c / det, -b / det, a / det);
    let [u, v] = mean;
    Ok([
        (count / total as f64).ln()
            - 0.5 * det.ln()
            - 0.5 * (aa * u * u + 2. * bb * u * v + cc * v * v),
        aa * u + bb * v,
        bb * u + cc * v,
        -0.5 * aa,
        -bb,
        -0.5 * cc,
        0.,
        0.,
    ])
}
pub fn run(input: Input) -> Result<(Model, Report)> {
    let mut model = Model::zero(input)?;
    let (n, v, b) = (model.input.n, model.input.vocabulary, model.input.blocks);
    let bank = directions()?;
    let mut report = Report {
        blocks: vec![],
        searched_directions: 0,
        direction_records: 0,
        moment_records: 0,
        materialized_coefficients: model.coefficients.len(),
        projected_coefficients: model.coefficients.len(),
    };
    for block in 0..b {
        let mut groups = BTreeMap::<usize, Vec<&Record>>::new();
        for r in &model.input.records {
            if model.input.route[r.cell] == block {
                groups.entry(r.label).or_default().push(r);
            }
        }
        let count = groups.values().map(Vec::len).sum();
        let mut record = BlockReport {
            block,
            labels: groups.keys().copied().collect(),
            records: count,
            method: "empty_zero".into(),
            candidates: vec![],
            selected: None,
            margin_scale: None,
        };
        let mut polynomials = BTreeMap::<usize, [f64; STATES]>::new();
        if groups.len() == 1 {
            let mut a = [0.; STATES];
            a[0] = 192.;
            polynomials.insert(record.labels[0], a);
            record.method = "single_class".into();
        } else if groups.len() >= 2 {
            if groups.len() == 2 {
                record.candidates = bank.iter().map(|&d| candidate(n, &groups, d)).collect();
                report.searched_directions += bank.len();
                report.direction_records += bank.len() * count;
                for (i, d) in record.candidates.iter().enumerate() {
                    if d.gap > 1e-10
                        && record
                            .selected
                            .is_none_or(|j| d.gap > record.candidates[j].gap)
                    {
                        record.selected = Some(i);
                    }
                }
            }
            if let Some(index) = record.selected {
                let d = &record.candidates[index];
                let mut margin = f64::INFINITY;
                for rows in groups.values() {
                    for r in rows {
                        let [u, w] = features::point(n, r.cell);
                        let (_, norm) = features::row(n, r.cell);
                        margin =
                            margin.min((d.axis[0] * u + d.axis[1] * w - d.threshold).abs() / norm);
                    }
                }
                let scale = 2. / margin;
                let common = 192. + scale * (d.axis[0].abs() + d.axis[1].abs() + d.threshold.abs());
                for (label, sign) in [(d.lower, -1.), (d.upper, 1.)] {
                    polynomials.insert(
                        label,
                        [
                            common - sign * scale * d.threshold,
                            sign * scale * d.axis[0],
                            sign * scale * d.axis[1],
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                        ],
                    );
                }
                record.method = "separated_intervals".into();
                record.margin_scale = Some(scale);
            } else {
                for (&label, rows) in &groups {
                    polynomials.insert(label, moments(n, rows, count)?);
                    report.moment_records += rows.len();
                }
                let common = 192.
                    + polynomials
                        .values()
                        .map(|a| a.iter().map(|x| x.abs()).sum::<f64>())
                        .fold(0., f64::max);
                for a in polynomials.values_mut() {
                    a[0] += common;
                }
                record.method = "quadratic_moments".into();
            }
        }
        for (&label, a) in &polynomials {
            for f in 0..STATES {
                let at = (block * WIDTH + 2 * f) * v + label;
                for (i, &value) in a.iter().enumerate() {
                    let p = features::phase(i, f);
                    model.coefficients[at] += value * p.re;
                    model.coefficients[at + v] -= value * p.im;
                }
            }
        }
        report.blocks.push(record);
    }
    model.project();
    model.validate()?;
    Ok((model, report))
}
