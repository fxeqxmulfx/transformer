//! Direct joint tensor, ordinary CE, and the exact CC8 linear oracle.
//! Lean proves convexity, oracle optimality and feasible updates, but not
//! floating-point execution or convergence of this optimization loop.

use convex_recall_compare::{convex::ce_gradient, data::Sample};

pub struct Channel {
    pub queries: usize,
    pub keys: usize,
    pub values: usize,
    pub classes: usize,
    pub weights: Vec<f64>,
}

impl Channel {
    pub fn new(queries: usize, keys: usize, values: usize, classes: usize) -> Self {
        assert!([queries, keys, values, classes].iter().all(|&n| n > 0));
        Self {
            queries,
            keys,
            values,
            classes,
            weights: vec![1.0 / (keys * classes) as f64; queries * keys * values * classes],
        }
    }

    pub fn index(&self, q: usize, k: usize, v: usize, c: usize) -> usize {
        (((q * self.keys + k) * self.values + v) * self.classes) + c
    }

    /// Reads only the query and bindings. Targets are used only by loss_gradient.
    pub fn logits(&self, sample: &Sample) -> Vec<f64> {
        assert_eq!(sample.values.len(), self.keys);
        let mut z = vec![0.0; sample.queries.len() * self.classes];
        for (i, &q) in sample.queries.iter().enumerate() {
            assert!(q < self.queries);
            for (k, &v) in sample.values.iter().enumerate() {
                assert!(v < self.values);
                let at = self.index(q, k, v, 0);
                for c in 0..self.classes {
                    z[i * self.classes + c] += self.weights[at + c];
                }
            }
        }
        z
    }

    pub fn loss_gradient(&self, samples: &[&Sample]) -> (f64, Vec<f64>) {
        let count: usize = samples.iter().map(|s| s.targets.len()).sum();
        assert!(count > 0);
        let mut gradient = vec![0.0; self.weights.len()];
        let mut loss = 0.0;
        for &sample in samples {
            assert_eq!(sample.queries.len(), sample.targets.len());
            let z = self.logits(sample);
            for (i, &target) in sample.targets.iter().enumerate() {
                let (ce, dz) = ce_gradient(&z[i * self.classes..][..self.classes], target);
                loss += ce;
                let q = sample.queries[i];
                for (k, &v) in sample.values.iter().enumerate() {
                    let at = self.index(q, k, v, 0);
                    for c in 0..self.classes {
                        gradient[at + c] += dz[c];
                    }
                }
            }
        }
        for g in &mut gradient {
            *g /= count as f64;
        }
        (loss / count as f64, gradient)
    }

    /// Exact finite linear minimization, proved by ConvexChannel.linear_oracle.
    /// Each key's best output class is chosen independently at every input value.
    pub fn oracle(&self, gradient: &[f64]) -> Vec<f64> {
        assert_eq!(gradient.len(), self.weights.len());
        assert!(gradient.iter().all(|x| x.is_finite()));
        let mut vertex = vec![0.0; gradient.len()];
        for q in 0..self.queries {
            let mut best_key = 0;
            let mut best_cost = f64::INFINITY;
            let mut choices = vec![0; self.keys * self.values];
            for k in 0..self.keys {
                let mut cost = 0.0;
                for v in 0..self.values {
                    let at = self.index(q, k, v, 0);
                    let mut best_class = 0;
                    for c in 1..self.classes {
                        if gradient[at + c] < gradient[at + best_class] {
                            best_class = c;
                        }
                    }
                    choices[k * self.values + v] = best_class;
                    cost += gradient[at + best_class];
                }
                if cost < best_cost {
                    best_cost = cost;
                    best_key = k;
                }
            }
            for v in 0..self.values {
                let at = self.index(q, best_key, v, choices[best_key * self.values + v]);
                vertex[at] = 1.0;
            }
        }
        vertex
    }

    /// Full-batch Frank-Wolfe, eta=2/(step+1), step starting at one.
    pub fn step(&mut self, samples: &[&Sample], step: u64) {
        assert!(step > 0);
        let (_, gradient) = self.loss_gradient(samples);
        let vertex = self.oracle(&gradient);
        let eta = 2.0 / (step + 1) as f64;
        for (w, s) in self.weights.iter_mut().zip(vertex) {
            *w = (1.0 - eta) * *w + eta * s;
        }
    }

    /// Floating-point Frank-Wolfe gap on the supplied full training dataset.
    pub fn gap(&self, samples: &[&Sample]) -> f64 {
        let (_, gradient) = self.loss_gradient(samples);
        let vertex = self.oracle(&gradient);
        gradient.iter().zip(self.weights.iter().zip(vertex)).map(|(g, (w, s))| g * (w - s)).sum()
    }

    /// Maximum violation of nonnegativity, shared routing masses and unit mass.
    pub fn feasibility_error(&self) -> f64 {
        let mut error: f64 = 0.0;
        for &w in &self.weights {
            assert!(w.is_finite());
            error = error.max(-w);
        }
        for q in 0..self.queries {
            let mut total = 0.0;
            for k in 0..self.keys {
                let at = self.index(q, k, 0, 0);
                let reference: f64 = self.weights[at..at + self.classes].iter().sum();
                total += reference;
                for v in 1..self.values {
                    let at = self.index(q, k, v, 0);
                    let mass: f64 = self.weights[at..at + self.classes].iter().sum();
                    error = error.max((mass - reference).abs());
                }
            }
            error = error.max((total - 1.0).abs());
        }
        error
    }
}
