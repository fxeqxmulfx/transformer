//! EM1-EM4: checkpoint-only bounds for the committed plain Rotate90 graph.
//! Source: flat_approx/fixtures/reference.rs at 62638521; sqrt(2) product,
//! shared embedding, four RMSNorms with f32::EPSILON, no learned gains.
//! Logical hidden pairs use the native split real/imaginary buffer ordering.
//! These FP64 calculations are numerically tested, not outward-rounded proofs.
use std::ops::Range;

pub const EPSILON: f64 = f32::EPSILON as f64;

#[derive(Clone, Copy)]
pub struct Shape {
    pub classes: usize,
    pub width: usize,
    pub hidden_pairs: usize,
}
impl Shape {
    pub const ORIGINAL: Self = Self {
        classes: 97,
        width: 128,
        hidden_pairs: 341,
    };
    pub fn validate(self) {
        assert!(self.classes > 0 && self.width > 0 && self.width.is_multiple_of(2));
        assert!(self.hidden_pairs > 0);
    }
    pub fn hidden(self) -> usize {
        2 * self.hidden_pairs
    }
    pub fn embeddings(self) -> usize {
        self.classes * self.width
    }
    pub fn count(self) -> usize {
        self.embeddings() + 3 * self.width * self.hidden()
    }
    pub fn row(self, c: usize) -> Range<usize> {
        assert!(c < self.classes);
        c * self.width..(c + 1) * self.width
    }
    pub fn gate(self) -> Range<usize> {
        self.embeddings()..self.embeddings() + self.width * self.hidden()
    }
    pub fn value(self) -> Range<usize> {
        self.gate().end..self.gate().end + self.width * self.hidden()
    }
    pub fn output(self) -> Range<usize> {
        self.value().end..self.count()
    }
}

/// Euclidean/Frobenius displacement caps. The matrix cap also bounds its
/// operator norm; it includes every expanded entry of a tied source matrix.
pub struct Caps {
    pub rows: Vec<f64>,
    pub gate: f64,
    pub value: f64,
    pub output: f64,
}
impl Caps {
    fn validate(&self, shape: Shape) {
        assert_eq!(self.rows.len(), shape.classes);
        assert!(
            self.rows
                .iter()
                .chain([&self.gate, &self.value, &self.output])
                .all(|x| x.is_finite() && *x >= 0.0)
        );
    }
}

pub fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
fn dense(x: &[f64], a: &[f64]) -> Vec<f64> {
    assert_eq!(a.len() % x.len(), 0);
    a.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}
fn rms(x: &[f64]) -> Vec<f64> {
    let d = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + EPSILON).sqrt();
    x.iter().map(|v| v / d).collect()
}

/// Intersect the old-denominator bound, local secant and range diameter.
/// Source: FeatureMotion.rms_motion_radius. Zero and near-zero inputs are valid.
pub fn rms_radius(old: &[f64], radius: f64) -> f64 {
    assert!(!old.is_empty() && old.iter().all(|x| x.is_finite()));
    assert!(radius >= 0.0 && !radius.is_nan());
    let width = old.len() as f64;
    let s = (old.iter().map(|x| x * x).sum::<f64>() + width * EPSILON).sqrt();
    let mut bound = (2.0 * width.sqrt()).min(2.0 * width.sqrt() / s * radius);
    if radius < s {
        bound = bound.min(width.sqrt() * radius / (s * (s - radius)).sqrt());
    }
    bound
}

/// Old values and prospective stage radii; all fields are fixed at construction.
pub struct Stage {
    pub name: &'static str,
    pub old: Vec<f64>,
    pub radius: f64,
}
pub struct Forecast {
    pub stages: Vec<Stage>,
    pub old_logits: Vec<f64>,
    pub shape: Shape,
    old_readout: Vec<f64>,
    row_motion: Vec<f64>,
}
impl Forecast {
    pub fn feature(&self) -> &Stage {
        self.stages.last().unwrap()
    }
    /// PM3 plus EM4, with tied readout motion and its mixed feature term.
    /// An execution-contrast CHANGE allowance is separate from exact arithmetic.
    pub fn contrast_budget(&self, u: usize, c: usize, execution_change: f64) -> f64 {
        let contrast: Vec<_> = self.old_readout[self.shape.row(u)]
            .iter()
            .zip(&self.old_readout[self.shape.row(c)])
            .map(|(a, b)| a - b)
            .collect();
        if u == c {
            return 0.0;
        }
        crate::contrast_radius(
            norm(&contrast),
            norm(&self.feature().old),
            self.row_motion[u] + self.row_motion[c],
            self.feature().radius,
            execution_change,
        )
    }
    /// `old_execution[c]` is the observed old logit's residual relative to
    /// the exact source graph. `execution_change` bounds the change of each
    /// rival's residual contrast, rather than just its future residual.
    pub fn probability_ceiling(
        &self,
        u: usize,
        old_execution: &[f64],
        execution_change: f64,
    ) -> f64 {
        assert_eq!(old_execution.len(), self.shape.classes);
        let logits: Vec<_> = self
            .old_logits
            .iter()
            .zip(old_execution)
            .map(|(x, e)| x + e)
            .collect();
        let budgets: Vec<_> = (0..self.shape.classes)
            .map(|c| self.contrast_budget(u, c, execution_change))
            .collect();
        crate::rival_ceiling(&logits, u, &budgets)
    }
}

/// Source graph with no future state in its interface. Frobenius upper bounds
/// on old operators are conservative; no estimated singular values are used.
pub fn forecast(shape: Shape, old: &[f64], a: usize, b: usize, caps: &Caps) -> Forecast {
    shape.validate();
    caps.validate(shape);
    assert_eq!(old.len(), shape.count());
    assert!(old.iter().all(|x| x.is_finite()));
    assert!(a < shape.classes && b < shape.classes);
    let mut stages = Vec::new();
    let mut save = |name, values: &Vec<f64>, radius: f64| {
        assert!(radius.is_finite() && values.iter().all(|x| x.is_finite()));
        stages.push(Stage {
            name,
            old: values.clone(),
            radius,
        });
    };
    let gather: Vec<_> = (0..shape.width)
        .map(|k| {
            old[a * shape.width + k]
                + if k % 2 == 0 {
                    -old[b * shape.width + k + 1]
                } else {
                    old[b * shape.width + k - 1]
                }
        })
        .collect();
    let r0 = caps.rows[a] + caps.rows[b];
    save("gather", &gather, r0);
    let input = rms(&gather);
    let r1 = rms_radius(&gather, r0);
    save("embedding-rms", &input, r1);
    let inner = rms(&input);
    let r2 = rms_radius(&input, r1);
    save("pre-rms", &inner, r2);
    let gate = dense(&inner, &old[shape.gate()]);
    let value = dense(&inner, &old[shape.value()]);
    let rg = (norm(&old[shape.gate()]) + caps.gate) * r2 + caps.gate * norm(&inner);
    let ru = (norm(&old[shape.value()]) + caps.value) * r2 + caps.value * norm(&inner);
    save("gate", &gate, rg);
    save("value", &value, ru);
    let h = shape.hidden_pairs;
    let hidden: Vec<_> = (0..2 * h)
        .map(|k| {
            if k < h {
                (gate[k] * value[k] - gate[k + h] * value[k + h]) / 2.0_f64.sqrt()
            } else {
                let j = k - h;
                (gate[j] * value[j + h] + gate[j + h] * value[j]) / 2.0_f64.sqrt()
            }
        })
        .collect();
    let rh = (rg * norm(&value) + norm(&gate) * ru + rg * ru) / 2.0_f64.sqrt();
    save("product", &hidden, rh);
    let branch = dense(&hidden, &old[shape.output()]);
    let rb = (norm(&old[shape.output()]) + caps.output) * rh + caps.output * norm(&hidden);
    save("second-linear", &branch, rb);
    let normalized = rms(&branch);
    let rn = rms_radius(&branch, rb);
    save("branch-rms", &normalized, rn);
    let residual: Vec<_> = input.iter().zip(&normalized).map(|(x, y)| x + y).collect();
    save("residual", &residual, r1 + rn);
    let feature = rms(&residual);
    let rf = rms_radius(&residual, r1 + rn);
    save("final-rms", &feature, rf);
    Forecast {
        stages,
        old_logits: dense(&feature, &old[..shape.embeddings()]),
        shape,
        old_readout: old[..shape.embeddings()].to_vec(),
        row_motion: caps.rows.clone(),
    }
}
