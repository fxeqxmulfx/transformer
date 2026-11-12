//! Canonical observations, independent of arrival order and batch boundaries.
//! Repetitions are counted and deduplicated for the algebraic constraints.
//! Conflicting repetitions are rejected atomically, never silently replaced.
use crate::{Decoder, Observation, RUN_MEMORY_BYTES, Result, Shape, preflight};

#[derive(Clone, Debug)]
pub struct Training {
    pub(crate) shape: Shape,
    pub(crate) decoder: Decoder,
    labels: Vec<Option<usize>>,
    orbit_for_class: Vec<Option<usize>>,
    exposures: u64,
}

impl Training {
    /// CM18 needs one distinct output label per orbit, not just an even decoder.
    /// Unused output classes stay in the vocabulary. No input arithmetic is read.
    pub fn new(shape: Shape, decoder: Decoder) -> Result<Self> {
        preflight(shape, RUN_MEMORY_BYTES)?;
        decoder.validate(shape)?;
        let mut inverse = vec![None; shape.vocab];
        for (orbit, &class) in decoder.orbit_class.iter().enumerate() {
            if inverse[class].replace(orbit).is_some() {
                return Err("unsupported decoder: fibers must be exactly opposite pairs".into());
            }
        }
        Ok(Self {
            shape,
            decoder,
            labels: vec![None; shape.cells()],
            orbit_for_class: inverse,
            exposures: 0,
        })
    }

    /// Validate a whole batch before changing the retained observations.
    pub fn observe(&mut self, batch: &[Observation]) -> Result<()> {
        let exposures = self
            .exposures
            .checked_add(batch.len() as u64)
            .ok_or("observation counter overflow")?;
        let mut next = self.labels.clone();
        for &Observation { cell, label } in batch {
            if cell >= next.len() || label >= self.shape.vocab {
                return Err("observation outside the categorical table or vocabulary".into());
            }
            if self.orbit_for_class[label].is_none() {
                return Err("inconsistent observations: a label has no latent preimage".into());
            }
            if next[cell].is_some_and(|old| old != label) {
                return Err("inconsistent observations: a cell has conflicting labels".into());
            }
            next[cell] = Some(label);
        }
        self.labels = next;
        self.exposures = exposures;
        Ok(())
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn decoder(&self) -> &Decoder {
        &self.decoder
    }

    pub fn exposures(&self) -> u64 {
        self.exposures
    }

    pub fn unique_observations(&self) -> Vec<Observation> {
        self.labels
            .iter()
            .enumerate()
            .filter_map(|(cell, &label)| label.map(|label| Observation { cell, label }))
            .collect()
    }

    pub fn at(&self, row: usize, column: usize) -> Option<usize> {
        self.labels[row + self.shape.side * column]
    }

    pub(crate) fn states(&self, label: usize) -> Vec<usize> {
        match self.orbit_for_class[label] {
            Some(0) => vec![0],
            Some(orbit) => vec![orbit, self.shape.latent_states() - orbit],
            None => vec![],
        }
    }
}
