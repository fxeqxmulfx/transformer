//! Training-only constructors. Global moments retain block-constant descriptors;
//! unlike331 they are NOT centered separately in each block. Interval regression
//! shares noisy observed boundary midpoints through the same fixed geometry.
//! These analytic constructors perform no CE gradient, including forward mode.
use crate::*;
pub use convex_stress_shared_geometry::fit::{Score, fold};
use features::{Bank, Encoder};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Moments,
    Intervals,
    Tree1,
    Tree2,
}
pub const KINDS: [Kind; 4] = [Kind::Moments, Kind::Intervals, Kind::Tree1, Kind::Tree2];
impl Kind {
    pub fn depth(self) -> usize {
        match self {
            Self::Tree1 => 1,
            Self::Tree2 => 2,
            _ => 0,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub kind: Kind,
    pub ridge: f64,
    pub direction: usize,
    pub geometry: geometry::Geometry,
    pub partition: partition::Partition,
    pub split_attempts: Vec<partition::Attempt>,
    pub spatial: Vec<f64>,
    pub interval_gaps: Vec<Option<f64>>,
    pub decoder: Vec<Vec<usize>>,
    pub theta: Vec<f64>,
}
impl State {
    pub fn model(&self, input: &Input, gain: f64, common: f64) -> Result<Model> {
        let mut model = Model {
            input: input.clone(),
            protocol_sha256: protocol_hash()?,
            bank: Bank::Quadratic,
            geometry: self.geometry.clone(),
            partition: self.partition.clone(),
            common,
            decoder: self.decoder.clone(),
            theta: self.theta.iter().map(|x| x * gain).collect(),
        };
        model.radial()?;
        Ok(model)
    }
}
struct Group {
    labels: Vec<usize>,
    count: [usize; 2],
    point: [[f64; 2]; 2],
    cross: Vec<f64>,
}
struct Constructor {
    geometry: geometry::Geometry,
    groups: Vec<Group>,
    observations: Vec<(usize, usize, Vec<f64>)>,
    covariance: Vec<f64>,
    interval_covariance: Vec<f64>,
    records: usize,
    binary: usize,
}
pub(crate) fn factor(a: &[f64], dimension: usize, ridge: f64) -> Result<Vec<f64>> {
    if !RIDGES.contains(&ridge) || a.len() != dimension * dimension {
        return Err("invalid covariance or prospective ridge".into());
    }
    let mut l = vec![0.; a.len()];
    for j in 0..dimension {
        for k in 0..=j {
            let mut v = a[j * dimension + k] + if j == k { ridge } else { 0. };
            for h in 0..k {
                v -= l[j * dimension + h] * l[k * dimension + h];
            }
            l[j * dimension + k] = if j == k {
                if !v.is_finite() || v <= 0. {
                    return Err("nonpositive covariance factor".into());
                }
                v.sqrt()
            } else {
                v / l[k * dimension + k]
            };
        }
    }
    Ok(l)
}
pub(crate) fn solve(l: &[f64], mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let d = rhs.len();
    for j in 0..d {
        for k in 0..j {
            rhs[j] -= l[j * d + k] * rhs[k];
        }
        rhs[j] /= l[j * d + j];
    }
    for j in (0..d).rev() {
        for k in j + 1..d {
            rhs[j] -= l[k * d + j] * rhs[k];
        }
        rhs[j] /= l[j * d + j];
    }
    if rhs.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite linear solve".into());
    }
    Ok(rhs)
}
impl Constructor {
    fn new(input: &Input, geometry: geometry::Geometry, indices: &[usize]) -> Result<Self> {
        let d = 6 + geometry.bank.dimension();
        let mut groups: Vec<_> = (0..input.blocks)
            .map(|_| Group {
                labels: vec![],
                count: [0; 2],
                point: [[0.; 2]; 2],
                cross: vec![0.; d],
            })
            .collect();
        if indices.is_empty() || indices.iter().any(|&i| i >= input.records.len()) {
            return Err("empty or invalid fitting records".into());
        }
        let mut observations = vec![];
        for &i in indices {
            let r = &input.records[i];
            let b = input.route[r.cell];
            let g = &mut groups[b];
            if !g.labels.contains(&r.label) {
                g.labels.push(r.label);
            }
            if g.labels.len() > 2 {
                return Err(
                    "boundary constructor requires at most two categorical classes per block"
                        .into(),
                );
            }
            let c = g.labels.iter().position(|&c| c == r.label).unwrap();
            g.count[c] += 1;
            let mut x = features::raw(Bank::Quadratic, input.n, r.cell);
            g.point[c][0] += x[1];
            g.point[c][1] += x[2];
            x.extend_from_slice(&geometry.descriptors[b]);
            observations.push((b, c, x));
        }
        for g in &mut groups {
            for c in 0..g.labels.len() {
                for x in &mut g.point[c] {
                    *x /= g.count[c] as f64;
                }
            }
        }
        let mut covariance = vec![0.; d * d];
        let mut records = 0usize;
        for (b, c, x) in &observations {
            let g = &mut groups[*b];
            if g.labels.len() != 2 {
                continue;
            }
            records += 1;
            let sign = if *c == 0 { -1. } else { 1. };
            for j in 0..d {
                g.cross[j] += sign * x[j];
                for k in 0..d {
                    covariance[j * d + k] += x[j] * x[k];
                }
            }
        }
        let h = geometry.bank.dimension();
        let binary = groups.iter().filter(|g| g.labels.len() == 2).count();
        if records == 0 || binary == 0 {
            return Err("no binary records for shared boundary fit".into());
        }
        for x in &mut covariance {
            *x /= records as f64;
        }
        let mut interval_covariance = vec![0.; h * h];
        for (b, g) in groups.iter().enumerate() {
            if g.labels.len() != 2 {
                continue;
            }
            let z = &geometry.descriptors[b];
            for j in 0..h {
                for k in 0..h {
                    interval_covariance[j * h + k] += z[j] * z[k] / binary as f64;
                }
            }
        }
        Ok(Self {
            geometry,
            groups,
            observations,
            covariance,
            interval_covariance,
            records,
            binary,
        })
    }
    fn state(&self, kind: Kind, ridge: f64, direction: usize) -> Result<State> {
        let dir = *DIRECTIONS.get(direction).ok_or("invalid orientation")?;
        let signs: Vec<_> = self
            .groups
            .iter()
            .map(|g| {
                let delta = dir[0] * (g.point[1][0] - g.point[0][0])
                    + dir[1] * (g.point[1][1] - g.point[0][1]);
                if g.labels.len() == 2 && delta < 0. {
                    -1.
                } else {
                    1.
                }
            })
            .collect();
        let decoder: Vec<_> = self
            .groups
            .iter()
            .zip(&signs)
            .map(|(g, &s)| {
                let mut labels = g.labels.clone();
                if labels.len() == 2 && s < 0. {
                    labels.swap(0, 1);
                }
                labels
            })
            .collect();
        let h = self.geometry.bank.dimension();
        let mut gaps = vec![None; self.groups.len()];
        let mut partition = partition::Partition::whole();
        let mut split_attempts = vec![];
        let spatial = if kind == Kind::Moments {
            let d = 6 + h;
            let mut rhs = vec![0.; d];
            for (g, sign) in self.groups.iter().zip(&signs) {
                if g.labels.len() == 2 {
                    for (a, x) in rhs.iter_mut().zip(&g.cross) {
                        *a += sign * x / self.records as f64;
                    }
                }
            }
            solve(&factor(&self.covariance, d, ridge)?, rhs)?
        } else {
            let mut lower = vec![f64::NEG_INFINITY; self.groups.len()];
            let mut upper = vec![f64::INFINITY; self.groups.len()];
            for (b, c, x) in &self.observations {
                if self.groups[*b].labels.len() != 2 {
                    continue;
                }
                let q = dir[0] * x[1] + dir[1] * x[2];
                if (*c == 0) == (signs[*b] > 0.) {
                    lower[*b] = lower[*b].max(q);
                } else {
                    upper[*b] = upper[*b].min(q);
                }
            }
            let mut rhs = vec![0.; h];
            let mut points = vec![];
            for (b, g) in self.groups.iter().enumerate() {
                if g.labels.len() != 2 {
                    continue;
                }
                let gap = upper[b] - lower[b];
                gaps[b] = Some(gap);
                let boundary = if gap > 0. {
                    (lower[b] + upper[b]) / 2.
                } else {
                    (dir[0] * (g.point[0][0] + g.point[1][0])
                        + dir[1] * (g.point[0][1] + g.point[1][1]))
                        / 2.
                };
                for (a, z) in rhs.iter_mut().zip(&self.geometry.descriptors[b]) {
                    *a -= boundary * z / self.binary as f64;
                }
                points.push(partition::Point {
                    block: b,
                    descriptor: self.geometry.descriptors[b].clone(),
                    target: -boundary,
                });
            }
            let eta = if kind.depth() == 0 {
                solve(&factor(&self.interval_covariance, h, ridge)?, rhs)?
            } else {
                let (tree, eta, attempts) = partition::fit(&points, ridge, kind.depth())?;
                partition = tree;
                split_attempts = attempts;
                eta
            };
            let mut raw = vec![0., dir[0], dir[1], 0., 0., 0.];
            raw.extend(eta);
            raw
        };
        let mut theta = features::conjugate(Bank::Quadratic, &spatial[..6]);
        theta.extend_from_slice(&spatial[6..]);
        Ok(State {
            kind,
            ridge,
            direction,
            geometry: self.geometry.clone(),
            partition,
            split_attempts,
            spatial,
            interval_gaps: gaps,
            decoder,
            theta,
        })
    }
}

pub fn score(model: &Model, rows: &[features::Row], indices: &[usize]) -> Score {
    let mut out = Score::default();
    for &i in indices {
        let row = &rows[i];
        let (loss, prediction) = model.compact_score(row);
        out.ce_sum += loss;
        out.correct += usize::from(prediction == row.label);
        out.records += 1;
    }
    out
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GainScore {
    pub gain: f64,
    pub common: f64,
    pub score: Score,
    pub model_sha256: String,
    pub weighted_mass: f64,
    pub complete_mass: f64,
    pub product_residual: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub fold: usize,
    pub frame: geometry::Frame,
    pub bank: geometry::Bank,
    pub kind: Kind,
    pub ridge: f64,
    pub direction: usize,
    pub state: Option<State>,
    pub refusal: Option<String>,
    pub gains: Vec<GainScore>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub fold_record_ids: Vec<Vec<usize>>,
    pub candidates: Vec<Candidate>,
    pub local_controls: Vec<convex_stress_shared_geometry::fit::Control>,
    pub selected: usize,
    pub pooled_scores: Vec<Option<Score>>,
    pub refit: State,
    pub gain: f64,
    pub common: f64,
    pub conditional_on_inherited_route: bool,
    pub ce_gradients: usize,
}
pub fn configuration_count() -> usize {
    geometry::FRAMES.len()
        * geometry::BANKS.len()
        * KINDS.len()
        * RIDGES.len()
        * DIRECTIONS.len()
        * GAINS.len()
        * COMMONS.len()
}
pub fn run(input: &Input) -> Result<(Model, Report)> {
    let mut candidates = vec![];
    let mut local_controls = vec![];
    let folds: Vec<Vec<_>> = (0..3)
        .map(|f| {
            (0..input.records.len())
                .filter(|&i| fold(input.records[i].cell) == f)
                .collect()
        })
        .collect();
    if folds.iter().any(Vec::is_empty) {
        return Err("three nonempty cell exclusion folds required".into());
    }
    let mut pooled = vec![Some(Score::default()); configuration_count()];
    let rows = Encoder::new(Bank::Quadratic).records(input);
    for (f, held) in folds.iter().enumerate() {
        let train: Vec<_> = (0..input.records.len())
            .filter(|&i| fold(input.records[i].cell) != f)
            .collect();
        let mut local_input = input.clone();
        local_input.records = train.iter().map(|&i| input.records[i].clone()).collect();
        let (local, _) = convex_stress_shared_head::initialize::run(local_input)?;
        let mut control = Score::default();
        for &i in held {
            let r = &input.records[i];
            let z = local.logits_cell(r.cell)?;
            control.ce_sum += cross_entropy(&z, r.label);
            control.records += 1;
            let prediction = (0..z.len())
                .max_by(|&a, &b| z[a].total_cmp(&z[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            control.correct += usize::from(prediction == r.label);
        }
        local_controls.push(convex_stress_shared_geometry::fit::Control {
            fold: f,
            model: local,
            score: control,
        });
        let mut at = 0;
        for frame in geometry::FRAMES {
            for bank in geometry::BANKS {
                let constructor = geometry::Geometry::new(input, frame, bank)
                    .and_then(|g| Constructor::new(input, g, &train));
                for kind in KINDS {
                    for ridge in RIDGES {
                        for direction in 0..DIRECTIONS.len() {
                            let fit = match &constructor {
                                Ok(c) => c.state(kind, ridge, direction),
                                Err(e) => Err(e.to_string().into()),
                            };
                            let mut candidate = Candidate {
                                fold: f,
                                frame,
                                bank,
                                kind,
                                ridge,
                                direction,
                                state: None,
                                refusal: None,
                                gains: vec![],
                            };
                            match fit {
                                Ok(state) => {
                                    for gain in GAINS {
                                        for common in COMMONS {
                                            let model = state.model(input, gain, common)?;
                                            let result = score(&model, &rows, held);
                                            if let Some(p) = &mut pooled[at] {
                                                p.ce_sum += result.ce_sum;
                                                p.correct += result.correct;
                                                p.records += result.records;
                                            }
                                            let (weights, _) = model.weights();
                                            candidate.gains.push(GainScore {
                                                gain,
                                                common,
                                                score: result,
                                                model_sha256: digest(&model)?,
                                                weighted_mass: projection::mass(
                                                    &model.theta,
                                                    &weights,
                                                ),
                                                complete_mass: model.expanded_mass(),
                                                product_residual: training::audit(
                                                    &model,
                                                    input.records[held[0]].cell,
                                                )?,
                                            });
                                            at += 1;
                                        }
                                    }
                                    candidate.state = Some(state);
                                }
                                Err(e) => {
                                    candidate.refusal = Some(e.to_string());
                                    for _ in 0..GAINS.len() * COMMONS.len() {
                                        pooled[at] = None;
                                        at += 1;
                                    }
                                }
                            }
                            candidates.push(candidate);
                        }
                    }
                }
            }
        }
    }
    let selected = pooled
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
        .min_by(|(i, a), (j, b)| a.ce_sum.total_cmp(&b.ce_sum).then_with(|| i.cmp(j)))
        .ok_or("all declared boundary constructors refused")?
        .0;
    let common = COMMONS[selected % COMMONS.len()];
    let gain = GAINS[(selected / COMMONS.len()) % GAINS.len()];
    let choice = &candidates[selected / (GAINS.len() * COMMONS.len())];
    let geometry = geometry::Geometry::new(input, choice.frame, choice.bank)?;
    let constructor = Constructor::new(
        input,
        geometry,
        &(0..input.records.len()).collect::<Vec<_>>(),
    )?;
    let refit = constructor.state(choice.kind, choice.ridge, choice.direction)?;
    let model = refit.model(input, gain, common)?;
    Ok((
        model,
        Report {
            fold_record_ids: folds
                .iter()
                .map(|ids| ids.iter().map(|&i| input.records[i].id).collect())
                .collect(),
            candidates,
            local_controls,
            selected,
            pooled_scores: pooled,
            refit,
            gain,
            common,
            conditional_on_inherited_route: true,
            ce_gradients: 0,
        },
    ))
}
