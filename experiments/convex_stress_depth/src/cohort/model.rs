use super::cache::Observations;
use crate::{
    Result, checkpoint,
    features::Bank,
    head::Head,
    independent::{self, Independent},
    metrics::Predictor,
};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

pub enum Model {
    Tensor(Head),
    Factor(Independent),
}
impl Model {
    pub fn bank(&self) -> &Bank {
        match self {
            Self::Tensor(m) => &m.bank,
            Self::Factor(m) => &m.bank,
        }
    }
    pub fn radius(&self) -> f64 {
        match self {
            Self::Tensor(m) => m.radius,
            Self::Factor(m) => m.radius,
        }
    }
    pub fn parameters(&self) -> usize {
        match self {
            Self::Tensor(m) => m.coefficients.len(),
            Self::Factor(m) => m.decoder.len() + m.probabilities.len(),
        }
    }
    pub fn gradient_size(&self) -> usize {
        match self {
            Self::Tensor(m) => m.coefficients.len(),
            Self::Factor(m) => m.decoder.len(),
        }
    }
    pub fn validate(&self) -> Result<f64> {
        match self {
            Self::Tensor(m) => m.validate(),
            Self::Factor(m) => m.validate(),
        }
    }
    pub fn objective(&self, cache: &mut Observations, order: &[usize]) -> Result<f64> {
        Ok(match self {
            Self::Tensor(m) => cache.objective(m, order, None)?.0,
            Self::Factor(m) => match cache {
                Observations::Dense(c) => independent::objective(m, c, order, None, None)?.0,
                Observations::Stream(_) => {
                    return Err("independent factors need the original bank".into());
                }
            },
        })
    }
    pub fn update(
        &mut self,
        cache: &mut Observations,
        batch: &[usize],
        scratch: &mut [f64],
    ) -> Result<(usize, usize)> {
        match self {
            Self::Tensor(m) => {
                let (_, n) = cache.objective(m, batch, Some(scratch))?;
                crate::head::gradient_step(m, scratch, 1. / cache.width() as f64)?;
                Ok((n, 0))
            }
            Self::Factor(m) => match cache {
                Observations::Dense(c) => independent::step(m, c, batch, scratch),
                Observations::Stream(_) => Err("independent factors need the original bank".into()),
            },
        }
    }
    pub fn gradient_passes(&self) -> usize {
        match self {
            Self::Factor(m) if m.bank.depth > 0 => 2,
            _ => 1,
        }
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        let output = BufWriter::new(File::create(path)?);
        match self {
            Self::Tensor(m) => checkpoint::save(m, output),
            Self::Factor(m) => checkpoint::save_independent(m, output),
        }
    }
    pub fn load(architecture: &str, path: &Path) -> Result<Self> {
        let input = BufReader::new(File::open(path)?);
        if architecture == "independent" {
            Ok(Self::Factor(checkpoint::load_independent(input)?))
        } else if ["joint", "additive"].contains(&architecture) {
            Ok(Self::Tensor(checkpoint::load(input)?))
        } else {
            Err("unknown saved architecture".into())
        }
    }
}
impl Predictor for Model {
    fn vocab(&self) -> usize {
        self.bank().vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        match self {
            Self::Tensor(m) => m.logits(prefix),
            Self::Factor(m) => m.logits(prefix),
        }
    }
}
