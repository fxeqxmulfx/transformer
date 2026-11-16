//! Complete raw object prefixes and their unchanged activation inequalities.
//!
//! Source: Lean `LearnedHinge.Projection.rawRow_dot`, `signedRow_dot`, and
//! CM76's complete parameter count. Object integers are serialization indices;
//! row values describe which learned coordinates the raw input reads. All
//! prefixes and scalar masks are counted before any model-domain allocation.

use crate::{ProjectionLimits, Rational, Refusal, projection::ProjectionProblem};
use num_traits::{One, Zero};

#[derive(Clone, Copy, Debug)]
pub struct DomainLimits {
    pub max_prefixes: usize,
    pub max_patterns: usize,
    pub max_real_parameters: usize,
    pub max_row_entries: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainPlan {
    pub objects: usize,
    pub context: usize,
    pub prefixes: usize,
    pub coordinates: usize,
    pub patterns: usize,
    pub real_parameters: usize,
}

impl DomainPlan {
    /// Count the whole declared family. An oversized family is refused before
    /// constructing rows; it is never replaced by sampled prefixes or masks.
    pub fn new(objects: usize, context: usize, limits: DomainLimits) -> Result<Self, Refusal> {
        if objects == 0 {
            return Err(Refusal::EmptyObjects);
        }
        let coordinates = context
            .checked_mul(objects)
            .and_then(|x| x.checked_add(1))
            .ok_or(Refusal::Overflow)?;
        let mut prefixes = 0usize;
        let mut at_length = 1usize;
        for length in 0..=context {
            prefixes = prefixes.checked_add(at_length).ok_or(Refusal::Overflow)?;
            if prefixes > limits.max_prefixes {
                return Err(Refusal::Budget);
            }
            if length < context {
                at_length = at_length.checked_mul(objects).ok_or(Refusal::Overflow)?;
            }
        }
        let patterns = 1usize
            .checked_shl(u32::try_from(prefixes).map_err(|_| Refusal::Overflow)?)
            .ok_or(Refusal::Overflow)?;
        let real_parameters = 4usize
            .checked_mul(objects)
            .and_then(|x| x.checked_mul(coordinates))
            .and_then(|x| x.checked_mul(patterns))
            .ok_or(Refusal::Overflow)?;
        let row_entries = prefixes.checked_mul(coordinates).ok_or(Refusal::Overflow)?;
        if patterns > limits.max_patterns
            || real_parameters > limits.max_real_parameters
            || row_entries > limits.max_row_entries
        {
            return Err(Refusal::Budget);
        }
        Ok(Self {
            objects,
            context,
            prefixes,
            coordinates,
            patterns,
            real_parameters,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RawDomain {
    plan: DomainPlan,
    prefixes: Vec<Vec<usize>>,
    rows: Vec<Vec<Rational>>,
}

impl RawDomain {
    pub fn new(objects: usize, context: usize, limits: DomainLimits) -> Result<Self, Refusal> {
        let plan = DomainPlan::new(objects, context, limits)?;
        let mut prefixes = vec![Vec::new()];
        let mut previous = 0..1;
        for _ in 0..context {
            let first = prefixes.len();
            for index in previous {
                for object in 0..objects {
                    let mut next = prefixes[index].clone();
                    next.push(object);
                    prefixes.push(next);
                }
            }
            previous = first..prefixes.len();
        }
        if prefixes.len() != plan.prefixes {
            return Err(Refusal::Shape);
        }
        let rows = prefixes
            .iter()
            .map(|tokens| {
                let mut row = vec![Rational::zero(); plan.coordinates];
                row[0] = Rational::one();
                for (position, &object) in tokens.iter().enumerate() {
                    row[1 + position * objects + object] = Rational::one();
                }
                row
            })
            .collect();
        Ok(Self {
            plan,
            prefixes,
            rows,
        })
    }

    pub fn plan(&self) -> &DomainPlan {
        &self.plan
    }
    pub fn prefixes(&self) -> &[Vec<usize>] {
        &self.prefixes
    }
    pub fn rows(&self) -> &[Vec<Rational>] {
        &self.rows
    }

    pub fn input_index(&self, tokens: &[usize]) -> Result<usize, Refusal> {
        if tokens.len() > self.plan.context {
            return Err(Refusal::ContextTooLong);
        }
        if tokens.iter().any(|&object| object >= self.plan.objects) {
            return Err(Refusal::InvalidObject);
        }
        // The complete admitted universe is present. No unknown-input fallback.
        self.prefixes
            .iter()
            .position(|prefix| prefix == tokens)
            .ok_or(Refusal::Shape)
    }

    /// One full oriented cone. The same scalar mask applies independently to
    /// both components in CM76; the later whole-model update must retain both.
    pub fn cone(
        &self,
        mask: usize,
        limits: ProjectionLimits,
    ) -> Result<ProjectionProblem, Refusal> {
        if mask >= self.plan.patterns {
            return Err(Refusal::InvalidPattern);
        }
        let normals = self
            .rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                row.iter()
                    .map(|entry| {
                        if mask & (1usize << index) != 0 {
                            entry.clone()
                        } else {
                            -entry
                        }
                    })
                    .collect()
            })
            .collect();
        ProjectionProblem::new(self.plan.coordinates, normals, limits)
    }
}
