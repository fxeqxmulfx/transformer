//! Original in-memory and retained-path streamed observations share one training
//! loop. Independent factors remain confined to the original composed bank.
use crate::{
    Result, data,
    features::Bank,
    head::{self, Cache, ForwardChoice, Head},
    stream_cache::{self, StreamCache},
};
use std::path::Path;

pub enum Observations {
    Dense(Cache),
    Stream(StreamCache),
}
impl Observations {
    pub fn build(bank: &Bank, data: &[data::Example], file: Option<&Path>) -> Result<Self> {
        Ok(match file {
            Some(path) => Self::Stream(StreamCache::build(path, bank, data)?),
            None => Self::Dense(Cache::build(bank, data)?),
        })
    }
    pub fn full_order(&self) -> Vec<usize> {
        match self {
            Self::Dense(c) => c.full_order(),
            Self::Stream(c) => c.full_order(),
        }
    }
    pub fn width(&self) -> usize {
        match self {
            Self::Dense(c) => c.width,
            Self::Stream(c) => c.width(),
        }
    }
    pub fn prefixes(&self, examples: &[usize]) -> Result<usize> {
        match self {
            Self::Dense(c) => examples.iter().try_fold(0, |n, &e| {
                if e >= c.example_offsets.len() - 1 {
                    Err("invalid example index".into())
                } else {
                    Ok(n + c.example_offsets[e + 1] - c.example_offsets[e])
                }
            }),
            Self::Stream(c) => c.prefixes(examples),
        }
    }
    pub fn initialize(
        &mut self,
        model: &mut Head,
        scratch: &mut [f64],
        choice: ForwardChoice,
    ) -> Result<usize> {
        match self {
            Self::Dense(c) => {
                if !matches!(choice, ForwardChoice::MinimumCe) {
                    return Err("original dense comparison keeps its frozen initializer".into());
                }
                head::forward_initialize(model, c, scratch)
            }
            Self::Stream(c) => {
                stream_cache::objective::forward_initialize(model, c, scratch, choice)
            }
        }
    }
    pub fn objective(
        &mut self,
        model: &Head,
        examples: &[usize],
        gradient: Option<&mut [f64]>,
    ) -> Result<(f64, usize)> {
        match self {
            Self::Dense(c) => head::objective(model, c, examples, gradient),
            Self::Stream(c) => stream_cache::objective::objective(model, c, examples, gradient),
        }
    }
}

#[cfg(test)]
mod tests;
