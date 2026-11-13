//! Actual CM32 complex block code and coefficient tensor. The table is an
//! optimization coordinate; prediction always multiplies saved complex weights.
use crate::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub observed: Vec<usize>,
    pub default_label: usize,
    pub offset: usize,
    pub code: Vec<C>,
}
impl Block {
    pub fn states(&self) -> usize {
        self.observed.len() + 1
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub n: usize,
    pub vocabulary: usize,
    pub rows: Vec<Option<u16>>,
    pub columns: Vec<Option<u16>>,
    pub blocks: Vec<Block>,
    pub table: Vec<f64>,
    /// Per frequency: vocabulary real coordinates, then vocabulary imaginary.
    pub coefficients: Vec<f64>,
    pub radius: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Encoded {
    pub block: usize,
    pub state: usize,
    pub label: usize,
}
pub(crate) fn phase(n: usize, state: usize, frequency: usize) -> C {
    C::from_polar(
        1. / (n as f64).sqrt(),
        std::f64::consts::TAU * ((state * frequency) % n) as f64 / n as f64,
    )
}
impl Model {
    pub fn bounds(&self) -> Result<limits::Bounds> {
        limits::dimensions(self.n, self.blocks.len(), self.vocabulary)?;
        limits::shape(
            self.vocabulary,
            &self.blocks.iter().map(Block::states).collect::<Vec<_>>(),
        )
    }
    pub fn latent(&self, cell: usize) -> usize {
        (self.rows[cell % self.n].unwrap_or(0) as usize
            + self.columns[cell / self.n].unwrap_or(0) as usize)
            % self.blocks.len()
    }
    pub fn known(&self, cell: usize) -> bool {
        cell < self.n * self.n
            && self.rows[cell % self.n].is_some()
            && self.columns[cell / self.n].is_some()
            && !self.blocks[self.latent(cell)].observed.is_empty()
    }
    pub fn location(&self, cell: usize) -> (usize, usize) {
        assert!(cell < self.n * self.n);
        let z = self.latent(cell);
        let b = &self.blocks[z];
        (z, b.observed.binary_search(&cell).unwrap_or(b.states() - 1))
    }
    pub fn logits_state(&self, block: usize, state: usize) -> Vec<f64> {
        let b = &self.blocks[block];
        assert!(state < b.states());
        let mut logits = vec![0.; self.vocabulary];
        for f in 0..b.states() {
            let phi = b.code[state * b.states() + f];
            let k = 2 * (b.offset + f) * self.vocabulary;
            for (c, value) in logits.iter_mut().enumerate() {
                *value += phi.re * self.coefficients[k + c]
                    - phi.im * self.coefficients[k + self.vocabulary + c];
            }
        }
        logits
    }
    pub fn logits_cell(&self, cell: usize) -> Vec<f64> {
        let (b, s) = self.location(cell);
        self.logits_state(b, s)
    }
    pub fn encode(&self, samples: &[[usize; 2]]) -> Result<Vec<Encoded>> {
        if samples.is_empty() || samples.len() > self.n * self.n {
            return Err("invalid refined observation count".into());
        }
        samples
            .iter()
            .map(|&[cell, label]| {
                if cell >= self.n * self.n || label >= self.vocabulary {
                    return Err("invalid observed categorical cell or label".into());
                }
                let (block, state) = self.location(cell);
                if state + 1 == self.blocks[block].states() {
                    return Err("training cell absent from fixed observed-input dictionary".into());
                }
                Ok(Encoded {
                    block,
                    state,
                    label,
                })
            })
            .collect()
    }
    /// Conjugate transform of all actual feasible table coordinates, CM32.
    pub fn materialize(&mut self) {
        self.coefficients.fill(0.);
        for b in &self.blocks {
            let n = b.states();
            for s in 0..n {
                let t = (b.offset + s) * self.vocabulary;
                for f in 0..n {
                    let p = b.code[s * n + f];
                    let h = 2 * (b.offset + f) * self.vocabulary;
                    for c in 0..self.vocabulary {
                        self.coefficients[h + c] += p.re * self.table[t + c];
                        self.coefficients[h + self.vocabulary + c] -= p.im * self.table[t + c];
                    }
                }
            }
        }
    }
    /// Coordinate clipping is a constrained update, not a learned activation.
    pub fn project(&mut self) {
        for b in &self.blocks {
            for s in 0..b.states() {
                let row = (b.offset + s) * self.vocabulary;
                for c in 0..self.vocabulary {
                    self.table[row + c] = if s + 1 == b.states() {
                        if c == b.default_label { 24. } else { 0. }
                    } else {
                        self.table[row + c].clamp(-32., 32.)
                    };
                }
            }
        }
        self.materialize();
    }
}
impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        assert!(prefix.len() >= 2);
        let (r, c) = (prefix[prefix.len() - 1], prefix[prefix.len() - 2]);
        assert!(r >= 0 && c >= 0 && (r as usize) < self.n && (c as usize) < self.n);
        self.logits_cell(r as usize + self.n * c as usize)
    }
}
