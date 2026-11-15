//! A shallow training-only regression tree for shared boundary coefficients.
//! Its descriptor thresholds and leaf assignment are frozen before convex CE.
//! CM55's fixed affine map charges every gated coefficient; tree discovery is
//! not asserted convex. Every attempted split and its refusal/loss is retained.
use crate::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Node {
    Leaf {
        slot: usize,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}
impl Node {
    pub fn leaf(&self, row: &[f64]) -> usize {
        match self {
            Self::Leaf { slot } => *slot,
            Self::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] <= *threshold {
                    left.leaf(row)
                } else {
                    right.leaf(row)
                }
            }
        }
    }
    fn validate(&self, dimension: usize, depth: usize, slots: &mut Vec<usize>) -> Result<()> {
        match self {
            Self::Leaf { slot } => slots.push(*slot),
            Self::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if depth >= 2 || *feature >= dimension || !threshold.is_finite() {
                    return Err("invalid frozen partition".into());
                }
                left.validate(dimension, depth + 1, slots)?;
                right.validate(dimension, depth + 1, slots)?;
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Partition {
    pub root: Node,
    pub leaves: usize,
}
impl Partition {
    pub fn whole() -> Self {
        Self {
            root: Node::Leaf { slot: 0 },
            leaves: 1,
        }
    }
    pub fn validate(&self, dimension: usize) -> Result<()> {
        let mut slots = vec![];
        self.root.validate(dimension, 0, &mut slots)?;
        if self.leaves > 4 || slots != (0..self.leaves).collect::<Vec<_>>() {
            return Err("invalid partition leaf slots".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub path: Vec<bool>,
    pub feature: usize,
    pub threshold: f64,
    pub left: usize,
    pub right: usize,
    pub parent_sse: f64,
    pub split_sse: Option<f64>,
    pub refusal: Option<String>,
}
#[derive(Clone, Debug)]
pub struct Point {
    pub block: usize,
    pub descriptor: Vec<f64>,
    pub target: f64,
}
#[derive(Clone)]
struct Stats {
    n: usize,
    d: usize,
    xx: Vec<f64>,
    xy: Vec<f64>,
    yy: f64,
}
impl Stats {
    fn zero(d: usize) -> Self {
        Self {
            n: 0,
            d,
            xx: vec![0.; d * d],
            xy: vec![0.; d],
            yy: 0.,
        }
    }
    fn add(&mut self, p: &Point, sign: f64) {
        if sign > 0. {
            self.n += 1;
        } else {
            self.n -= 1;
        }
        self.yy += sign * p.target * p.target;
        for j in 0..self.d {
            self.xy[j] += sign * p.descriptor[j] * p.target;
            for k in 0..self.d {
                self.xx[j * self.d + k] += sign * p.descriptor[j] * p.descriptor[k];
            }
        }
    }
    fn fit(&self, ridge: f64) -> Result<(Vec<f64>, f64)> {
        if self.n == 0 {
            return Err("empty partition leaf".into());
        }
        let a: Vec<_> = self.xx.iter().map(|x| x / self.n as f64).collect();
        let rhs = self.xy.iter().map(|x| x / self.n as f64).collect();
        let coef = fit::solve(&fit::factor(&a, self.d, ridge)?, rhs)?;
        let mut error = self.yy;
        for j in 0..self.d {
            error -= 2. * coef[j] * self.xy[j];
            for k in 0..self.d {
                error += coef[j] * coef[k] * self.xx[j * self.d + k];
            }
        }
        if !error.is_finite() || error < -1e-8 {
            return Err("invalid boundary residual".into());
        }
        Ok((coef, error.max(0.)))
    }
}
struct Builder<'a> {
    points: &'a [Point],
    ridge: f64,
    coefficients: Vec<Vec<f64>>,
    attempts: Vec<Attempt>,
}
impl Builder<'_> {
    fn node(&mut self, ids: &[usize], depth: usize, path: &[bool]) -> Result<Node> {
        let d = self.points[ids[0]].descriptor.len();
        let mut total = Stats::zero(d);
        for &i in ids {
            total.add(&self.points[i], 1.);
        }
        let (coefficients, parent_sse) = total.fit(self.ridge)?;
        let mut best: Option<(f64, usize, f64)> = None;
        if depth > 0 {
            for feature in 0..d {
                let mut order = ids.to_vec();
                order.sort_by(|&a, &b| {
                    self.points[a].descriptor[feature]
                        .total_cmp(&self.points[b].descriptor[feature])
                        .then_with(|| self.points[a].block.cmp(&self.points[b].block))
                });
                let mut left = Stats::zero(d);
                let mut right = total.clone();
                for k in 0..order.len() - 1 {
                    let p = &self.points[order[k]];
                    left.add(p, 1.);
                    right.add(p, -1.);
                    let x = p.descriptor[feature];
                    let y = self.points[order[k + 1]].descriptor[feature];
                    if x == y {
                        continue;
                    }
                    let threshold = (x + y) / 2.;
                    let mut trial = Attempt {
                        path: path.to_vec(),
                        feature,
                        threshold,
                        left: left.n,
                        right: right.n,
                        parent_sse,
                        split_sse: None,
                        refusal: None,
                    };
                    match left
                        .fit(self.ridge)
                        .and_then(|(_, a)| right.fit(self.ridge).map(|(_, b)| a + b))
                    {
                        Ok(error) => {
                            trial.split_sse = Some(error);
                            if error + 1e-12 < parent_sse
                                && best.as_ref().is_none_or(|(e, _, _)| error < *e)
                            {
                                best = Some((error, feature, threshold));
                            }
                        }
                        Err(e) => trial.refusal = Some(e.to_string()),
                    }
                    self.attempts.push(trial);
                }
            }
        }
        if let Some((_, feature, threshold)) = best {
            let (left, right): (Vec<_>, Vec<_>) = ids
                .iter()
                .copied()
                .partition(|&i| self.points[i].descriptor[feature] <= threshold);
            let mut lpath = path.to_vec();
            lpath.push(false);
            let mut rpath = path.to_vec();
            rpath.push(true);
            Ok(Node::Split {
                feature,
                threshold,
                left: Box::new(self.node(&left, depth - 1, &lpath)?),
                right: Box::new(self.node(&right, depth - 1, &rpath)?),
            })
        } else {
            let slot = self.coefficients.len();
            self.coefficients.push(coefficients);
            Ok(Node::Leaf { slot })
        }
    }
}
pub fn fit(
    points: &[Point],
    ridge: f64,
    depth: usize,
) -> Result<(Partition, Vec<f64>, Vec<Attempt>)> {
    if points.is_empty()
        || depth > 2
        || points[0].descriptor.is_empty()
        || points.iter().any(|p| {
            p.descriptor.len() != points[0].descriptor.len()
                || !p.target.is_finite()
                || p.descriptor.iter().any(|x| !x.is_finite())
        })
    {
        return Err("invalid boundary partition fitting inputs".into());
    }
    let mut builder = Builder {
        points,
        ridge,
        coefficients: vec![],
        attempts: vec![],
    };
    let root = builder.node(&(0..points.len()).collect::<Vec<_>>(), depth, &[])?;
    let partition = Partition {
        root,
        leaves: builder.coefficients.len(),
    };
    partition.validate(points[0].descriptor.len())?;
    Ok((
        partition,
        builder.coefficients.into_iter().flatten().collect(),
        builder.attempts,
    ))
}
