use crate::{
    Result, checkpoint,
    features::Bank,
    head::{Cache, Head},
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
    pub fn objective(&self, cache: &Cache, order: &[usize]) -> Result<f64> {
        Ok(match self {
            Self::Tensor(m) => crate::head::objective(m, cache, order, None)?.0,
            Self::Factor(m) => independent::objective(m, cache, order, None, None)?.0,
        })
    }
    pub fn update(
        &mut self,
        cache: &Cache,
        batch: &[usize],
        scratch: &mut [f64],
    ) -> Result<(usize, usize)> {
        match self {
            Self::Tensor(m) => {
                let (_, n) = crate::head::objective(m, cache, batch, Some(scratch))?;
                crate::head::gradient_step(m, scratch, 1. / cache.width as f64)?;
                Ok((n, 0))
            }
            Self::Factor(m) => independent::step(m, cache, batch, scratch),
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
