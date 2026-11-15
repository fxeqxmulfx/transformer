//! Analytic class-code moment constructor and conditional training folds.
//! No CE gradient occurs here. The inherited route is fixed from original
//! training; exclusion applies to fitting this head, not to route discovery.
use crate::*;
use features::{Bank, Encoder};

pub fn fold(cell: usize) -> usize {
    let mut x = (cell as u64).wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((x ^ (x >> 31)) % 3) as usize
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub bank: Bank,
    pub ridge: f64,
    pub direction: usize,
    /// Ridge0 denotes a declared fixed direction, without a covariance solve.
    pub calibrated: bool,
    pub interval_gaps: Vec<Option<f64>>,
    pub spatial: Vec<f64>,
    pub decoder: Vec<Vec<usize>>,
    pub theta: Vec<f64>,
}
impl State {
    pub fn model(&self, input: &Input, gain: f64) -> Result<Model> {
        let mut model = Model {
            input: input.clone(),
            protocol_sha256: protocol_hash()?,
            bank: self.bank,
            decoder: self.decoder.clone(),
            theta: self.theta.iter().map(|x| x * gain).collect(),
        };
        model.radial()?;
        Ok(model)
    }
}
struct Group {
    labels: Vec<usize>,
    count: usize,
    mean: Vec<f64>,
    mean_y: f64,
    cross: Vec<f64>,
    class_count: [usize; 2],
    class_point: [[f64; 2]; 2],
}
pub struct Moments {
    bank: Bank,
    groups: Vec<Group>,
    cov: Vec<f64>,
    count: usize,
    observations: Vec<(usize, usize, Vec<f64>)>,
}
impl Moments {
    pub fn new(input: &Input, bank: Bank, indices: &[usize]) -> Result<Self> {
        if indices.is_empty() || indices.iter().any(|&i| i >= input.records.len()) {
            return Err("empty or invalid constructor records".into());
        }
        let d = bank.dimension();
        let mut groups: Vec<_> = (0..input.blocks)
            .map(|_| Group {
                labels: vec![],
                count: 0,
                mean: vec![0.; d],
                mean_y: 0.,
                cross: vec![0.; d],
                class_count: [0; 2],
                class_point: [[0.; 2]; 2],
            })
            .collect();
        for &i in indices {
            let r = &input.records[i];
            let g = &mut groups[input.route[r.cell]];
            if !g.labels.contains(&r.label) {
                g.labels.push(r.label);
            }
            if g.labels.len() > 2 {
                return Err(
                    "shared binary geometry refuses a block with more than two raw classes".into(),
                );
            }
            let c = g.labels.iter().position(|&c| c == r.label).unwrap();
            g.count += 1;
            g.class_count[c] += 1;
            let point = convex_stress_shared_head::features::point(input.n, r.cell);
            for (s, x) in g.class_point[c].iter_mut().zip(point) {
                *s += x;
            }
            for (s, x) in g.mean.iter_mut().zip(features::raw(bank, input.n, r.cell)) {
                *s += x;
            }
            g.mean_y += if c == 0 { -1. } else { 1. };
        }
        for g in &mut groups {
            if g.count > 0 {
                for x in &mut g.mean {
                    *x /= g.count as f64;
                }
                g.mean_y /= g.count as f64;
                for c in 0..g.labels.len() {
                    for x in &mut g.class_point[c] {
                        *x /= g.class_count[c] as f64;
                    }
                }
            }
        }
        let count = groups
            .iter()
            .filter(|g| g.labels.len() == 2)
            .map(|g| g.count)
            .sum::<usize>();
        let mut cov = vec![0.; d * d];
        for &i in indices {
            let r = &input.records[i];
            let g = &mut groups[input.route[r.cell]];
            if g.labels.len() != 2 {
                continue;
            }
            let x: Vec<_> = features::raw(bank, input.n, r.cell)
                .iter()
                .zip(&g.mean)
                .map(|(x, m)| x - m)
                .collect();
            let y = (if r.label == g.labels[0] { -1. } else { 1. }) - g.mean_y;
            for j in 0..d {
                g.cross[j] += x[j] * y;
                for k in 0..=j {
                    cov[j * d + k] += x[j] * x[k];
                }
            }
        }
        let denominator = count.max(1) as f64;
        for j in 0..d {
            for k in 0..=j {
                cov[j * d + k] /= denominator;
                cov[k * d + j] = cov[j * d + k];
            }
        }
        Ok(Self {
            bank,
            groups,
            cov,
            count,
            observations: indices
                .iter()
                .map(|&i| {
                    let r = &input.records[i];
                    (
                        input.route[r.cell],
                        r.label,
                        features::raw(bank, input.n, r.cell),
                    )
                })
                .collect(),
        })
    }
    pub fn factor(&self, ridge: f64) -> Result<Vec<f64>> {
        if !ridge.is_finite() || ridge <= 0. {
            return Err("invalid positive covariance ridge".into());
        }
        let d = self.bank.dimension();
        let mut l = vec![0.; d * d];
        for j in 0..d {
            for k in 0..=j {
                let mut v = self.cov[j * d + k] + if j == k { ridge } else { 0. };
                for h in 0..k {
                    v -= l[j * d + h] * l[k * d + h];
                }
                l[j * d + k] = if j == k {
                    if !v.is_finite() || v <= 0. {
                        return Err("covariance factorization refused".into());
                    }
                    v.sqrt()
                } else {
                    v / l[k * d + k]
                };
            }
        }
        Ok(l)
    }
    pub fn solve(&self, ridge: f64, direction: usize, l: &[f64]) -> Result<State> {
        if direction >= DIRECTIONS.len() {
            return Err("invalid orientation direction".into());
        }
        let d = self.bank.dimension();
        let direction_row = DIRECTIONS[direction];
        let signs: Vec<_> = self
            .groups
            .iter()
            .map(|g| {
                let delta = (0..2)
                    .map(|j| direction_row[j] * (g.class_point[1][j] - g.class_point[0][j]))
                    .sum::<f64>();
                if g.labels.len() == 2 && delta < 0. {
                    -1.
                } else {
                    1.
                }
            })
            .collect();
        let mut a = vec![0.; d];
        for (g, &sign) in self.groups.iter().zip(&signs) {
            if g.labels.len() == 2 {
                for (a, x) in a.iter_mut().zip(&g.cross) {
                    *a += sign * x / self.count.max(1) as f64;
                }
            }
        }
        for j in 0..d {
            for k in 0..j {
                a[j] -= l[j * d + k] * a[k];
            }
            a[j] /= l[j * d + j];
        }
        for j in (0..d).rev() {
            for k in j + 1..d {
                a[j] -= l[k * d + j] * a[k];
            }
            a[j] /= l[j * d + j];
        }
        let mut theta = features::conjugate(self.bank, &a);
        let mut decoder = vec![];
        for (g, &sign) in self.groups.iter().zip(&signs) {
            let mut labels = g.labels.clone();
            if labels.len() == 2 && sign < 0. {
                labels.swap(0, 1);
            }
            theta.push(if labels.len() == 2 {
                sign * g.mean_y - a.iter().zip(&g.mean).map(|(a, m)| a * m).sum::<f64>()
            } else {
                0.
            });
            decoder.push(labels);
        }
        if theta.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite moment constructor".into());
        }
        Ok(State {
            bank: self.bank,
            ridge,
            direction,
            calibrated: false,
            interval_gaps: vec![None; self.groups.len()],
            spatial: a,
            decoder,
            theta,
        })
    }
    pub fn direct(&self, direction: usize) -> Result<State> {
        let dir = *DIRECTIONS.get(direction).ok_or("invalid fixed direction")?;
        let mut a = vec![0.; self.bank.dimension()];
        a[1] = dir[0];
        a[2] = dir[1];
        let mut theta = features::conjugate(self.bank, &a);
        let mut decoder = vec![];
        for g in &self.groups {
            let mut labels = g.labels.clone();
            let q: Vec<_> = g
                .class_point
                .iter()
                .map(|x| x[0] * dir[0] + x[1] * dir[1])
                .collect();
            if labels.len() == 2 && q[1] < q[0] {
                labels.swap(0, 1);
            }
            theta.push(if labels.len() == 2 {
                -(q[0] + q[1]) / 2.
            } else {
                0.
            });
            decoder.push(labels);
        }
        Ok(State {
            bank: self.bank,
            ridge: 0.,
            direction,
            calibrated: false,
            interval_gaps: vec![None; self.groups.len()],
            spatial: a,
            decoder,
            theta,
        })
    }
    /// Use the midpoint only when the projected training intervals separate.
    /// Overlap retains the moment intercept (or a balanced class-centroid
    /// intercept for fixed directions). No absent input/label enters this fit.
    pub fn calibrate(&self, mut state: State) -> State {
        let mut lower = vec![f64::NEG_INFINITY; self.groups.len()];
        let mut upper = vec![f64::INFINITY; self.groups.len()];
        for (b, label, x) in &self.observations {
            if state.decoder[*b].len() != 2 {
                continue;
            }
            let q = state.spatial.iter().zip(x).map(|(a, x)| a * x).sum::<f64>();
            if *label == state.decoder[*b][0] {
                lower[*b] = lower[*b].max(q);
            } else {
                upper[*b] = upper[*b].min(q);
            }
        }
        for (b, labels) in state.decoder.iter().enumerate() {
            if labels.len() == 2 {
                let gap = upper[b] - lower[b];
                state.interval_gaps[b] = Some(gap);
                if gap > 0. {
                    state.theta[2 * self.bank.states() + b] = -(lower[b] + upper[b]) / 2.;
                }
            }
        }
        state.calibrated = true;
        state
    }
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Score {
    pub ce_sum: f64,
    pub correct: usize,
    pub records: usize,
}
pub fn score(model: &Model, rows: &[features::Row], indices: &[usize]) -> Score {
    let mut score = Score::default();
    for &i in indices {
        let row = &rows[i];
        let z = model.logits_row(row);
        score.ce_sum += cross_entropy(&z, row.label);
        let predicted = (0..z.len())
            .max_by(|&a, &b| z[a].total_cmp(&z[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        score.correct += usize::from(predicted == row.label);
        score.records += 1;
    }
    score
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GainScore {
    pub gain: f64,
    pub score: Score,
    pub model_sha256: String,
    pub weighted_mass: f64,
    pub complete_mass: f64,
    pub product_residual: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub fold: usize,
    pub bank: Bank,
    pub ridge: f64,
    pub direction: usize,
    pub calibrated: bool,
    pub state: Option<State>,
    pub refusal: Option<String>,
    pub gains: Vec<GainScore>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Control {
    pub fold: usize,
    pub model: convex_stress_shared_head::Model,
    pub score: Score,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub fold_record_ids: Vec<Vec<usize>>,
    pub candidates: Vec<Candidate>,
    pub local_controls: Vec<Control>,
    pub selected: usize,
    pub pooled_scores: Vec<Option<Score>>,
    pub refit: State,
    pub gain: f64,
    pub conditional_on_inherited_route: bool,
    pub ce_gradients: usize,
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
        return Err("three nonempty input-exclusion folds required".into());
    }
    let count = features::BANKS.len() * (2 * RIDGES.len() + 1) * DIRECTIONS.len() * GAINS.len();
    let mut pooled: Vec<Option<Score>> = vec![Some(Score::default()); count];
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
            let predicted = (0..z.len())
                .max_by(|&a, &b| z[a].total_cmp(&z[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            control.correct += usize::from(predicted == r.label);
        }
        local_controls.push(Control {
            fold: f,
            model: local,
            score: control,
        });
        let mut at = 0;
        for bank in features::BANKS {
            let rows = Encoder::new(bank).records(input);
            let moments = Moments::new(input, bank, &train);
            for ridge in RIDGES.into_iter().chain([0.]) {
                let factor = match &moments {
                    Ok(_) if ridge == 0. => Ok(vec![]),
                    Ok(m) => m.factor(ridge),
                    Err(e) => Err(e.to_string().into()),
                };
                for direction in 0..DIRECTIONS.len() {
                    for calibrated in [false, true] {
                        if ridge == 0. && !calibrated {
                            continue;
                        }
                        let fit = match (&moments, &factor) {
                            (Ok(m), Ok(l)) => {
                                let state = if ridge == 0. {
                                    m.direct(direction)
                                } else {
                                    m.solve(ridge, direction, l)
                                };
                                state.map(|state| {
                                    if calibrated {
                                        m.calibrate(state)
                                    } else {
                                        state
                                    }
                                })
                            }
                            (_, Err(e)) => Err(e.to_string().into()),
                            (Err(e), _) => Err(e.to_string().into()),
                        };
                        let mut candidate = Candidate {
                            fold: f,
                            bank,
                            ridge,
                            direction,
                            calibrated,
                            state: None,
                            refusal: None,
                            gains: vec![],
                        };
                        match fit {
                            Ok(state) => {
                                for gain in GAINS {
                                    let model = state.model(input, gain)?;
                                    let score = score(&model, &rows, held);
                                    if let Some(p) = &mut pooled[at] {
                                        p.ce_sum += score.ce_sum;
                                        p.correct += score.correct;
                                        p.records += score.records;
                                    }
                                    let (weights, _) = model.weights();
                                    candidate.gains.push(GainScore {
                                        gain,
                                        score,
                                        model_sha256: digest(&model)?,
                                        weighted_mass: projection::mass(&model.theta, &weights),
                                        complete_mass: model.expanded_mass(),
                                        product_residual: training::audit(
                                            &model,
                                            input.records[held[0]].cell,
                                        )?,
                                    });
                                    at += 1;
                                }
                                candidate.state = Some(state);
                            }
                            Err(e) => {
                                candidate.refusal = Some(e.to_string());
                                for _ in GAINS {
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
    let selected = pooled
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
        .min_by(|(i, a), (j, b)| a.ce_sum.total_cmp(&b.ce_sum).then_with(|| i.cmp(j)))
        .ok_or("all declared constructors refused")?
        .0;
    let gain = GAINS[selected % GAINS.len()];
    let candidate = &candidates[selected / GAINS.len()];
    let all: Vec<_> = (0..input.records.len()).collect();
    let moments = Moments::new(input, candidate.bank, &all)?;
    let refit = if candidate.ridge == 0. {
        moments.direct(candidate.direction)?
    } else {
        moments.solve(
            candidate.ridge,
            candidate.direction,
            &moments.factor(candidate.ridge)?,
        )?
    };
    let refit = if candidate.calibrated {
        moments.calibrate(refit)
    } else {
        refit
    };
    let model = refit.model(input, gain)?;
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
            conditional_on_inherited_route: true,
            ce_gradients: 0,
        },
    ))
}
