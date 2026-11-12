use crate::*;
use propagation::Verified;

/// Input symbols, latent phases and output classes are separate dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub side: usize,
    pub states: usize,
    pub vocab: usize,
}

impl Shape {
    pub fn validate(self) -> Result<()> {
        if !(2..=256).contains(&self.side)
            || !(2..=512).contains(&self.states)
            || !(2..=1024).contains(&self.vocab)
        {
            return Err("unsupported input, latent or output dimensions".into());
        }
        Ok(())
    }
    pub fn cells(self) -> usize {
        self.side * self.side
    }
    pub fn entries(self) -> usize {
        self.cells() * self.states
    }
    pub fn cell(self, row: usize, column: usize) -> usize {
        row + self.side * column
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub shape: Shape,
    pub decoder: Vec<usize>,
    /// Cell-major full latent distributions, cells in column-major matrix order.
    pub latent: Vec<f64>,
    /// Original-vocabulary probability coordinates used as ordinary CE logits.
    pub output: Vec<f64>,
}

pub(crate) fn decode(shape: Shape, decoder: &[usize], latent: &[f64]) -> Vec<f64> {
    let mut out = vec![0.; shape.cells() * shape.vocab];
    for (row, target) in latent
        .chunks_exact(shape.states)
        .zip(out.chunks_exact_mut(shape.vocab))
    {
        for (&p, &y) in row.iter().zip(decoder) {
            target[y] += p;
        }
    }
    out
}

impl Model {
    pub fn new(shape: Shape, decoder: Vec<usize>, latent: Vec<f64>) -> Result<Self> {
        memory_bound(shape, 0, 0)?;
        if decoder.len() != shape.states
            || decoder.iter().any(|&y| y >= shape.vocab)
            || latent.len() != shape.entries()
        {
            return Err("invalid decoder or latent table size".into());
        }
        let output = decode(shape, &decoder, &latent);
        let result = Self {
            shape,
            decoder,
            latent,
            output,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<()> {
        memory_bound(self.shape, 0, 0)?;
        let s = self.shape;
        if self.decoder.len() != s.states
            || self.decoder.iter().any(|&y| y >= s.vocab)
            || self.latent.len() != s.entries()
            || self.output.len() != s.cells() * s.vocab
            || self
                .latent
                .iter()
                .any(|&x| !x.is_finite() || x < -1e-12 || x > 1. + 1e-12)
            || self
                .latent
                .chunks_exact(s.states)
                .any(|r| (r.iter().sum::<f64>() - 1.).abs() > 1e-10)
            || self.output != decode(s, &self.decoder, &self.latent)
        {
            return Err("invalid saved latent/output probability coordinates".into());
        }
        Ok(())
    }
    pub fn certified(&self, verified: &Verified) -> Result<()> {
        self.validate()?;
        let p = &verified.certificate().problem;
        if (self.shape.side, self.shape.states, self.shape.vocab) != (p.side, p.states, p.vocab)
            || !p.decoders.contains(&self.decoder)
        {
            return Err("model/certificate family mismatch".into());
        }
        for i in 0..p.side {
            for j in 0..p.side {
                if let Some(y) = verified.answer(i, j) {
                    let cell = self.shape.cell(i, j);
                    if self.output[cell * p.vocab..(cell + 1) * p.vocab]
                        .iter()
                        .enumerate()
                        .any(|(c, &x)| (x - f64::from(c == y)).abs() > 1e-12)
                    {
                        return Err("model violates a certified output equality".into());
                    }
                }
            }
        }
        Ok(())
    }
    /// CM19 branch_endpoint: choose the smallest retained phase in every domain
    /// of the first live branch for this decoder. No training-fit claim follows.
    pub fn anchor(verified: &Verified, index: usize) -> Result<Self> {
        let cert = verified.certificate();
        let p = &cert.problem;
        let shape = Shape {
            side: p.side,
            states: p.states,
            vocab: p.vocab,
        };
        memory_bound(shape, p.observations.len(), cert.memory_bound)?;
        let branch = cert
            .branches
            .iter()
            .find(|b| b.decoder == index && b.live)
            .ok_or("decoder has no surviving branch")?;
        let mut latent = vec![0.; shape.entries()];
        for i in 0..p.side {
            for j in 0..p.side {
                let phase = (branch.domains.rows[i][0] + branch.domains.columns[j][0]) % p.states;
                latent[shape.cell(i, j) * p.states + phase] = 1.;
            }
        }
        let model = Self::new(shape, p.decoders[index].clone(), latent)?;
        model.certified(verified)?;
        Ok(model)
    }
}
