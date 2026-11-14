//! CM41 at baf7c2e4: actual saved-parent routing with a partial input domain.
//! The checked CM38 heads and budgets are retained. A parent is evaluated only
//! inside its own domain; elsewhere the candidate's actual complex logits remain.
pub use convex_stress_fallback::{
    self as original, Binding, Bounds, Input, Parent, Result, causal,
};
use serde::{Deserialize, Serialize};

pub const PREDICTION_LIMIT: usize = 4096;
pub const ROUTING_POLICY: &str = "candidate_if_supported_or_parent_unavailable_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    CandidateSupported,
    CandidateParentUnavailable,
    Parent,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Route {
    pub candidate_supported: bool,
    pub parent_available: bool,
    pub source: Source,
}

#[derive(Clone, Debug, Serialize)]
pub struct Model {
    routing_policy: &'static str,
    original: original::Model,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    routing_policy: String,
    original: serde_json::Value,
}

impl Model {
    /// The original constructor binds actual support, both heads, original
    /// records and combined buffers. The new fixed gate adds no parameters.
    pub fn new(candidate: causal::model::Model, parent: Parent, input: &Input) -> Result<Self> {
        Ok(Self {
            routing_policy: ROUTING_POLICY,
            original: original::Model::new(candidate, parent, input)?,
        })
    }
    pub fn load(value: serde_json::Value, input: &Input, expected: &Binding) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if saved.routing_policy != ROUTING_POLICY {
            return Err("saved partial-parent routing policy changed".into());
        }
        Ok(Self {
            routing_policy: ROUTING_POLICY,
            original: original::Model::load(saved.original, input, expected)?,
        })
    }
    pub fn candidate(&self) -> &causal::model::Model {
        self.original.candidate()
    }
    pub fn parent(&self) -> &Parent {
        self.original.parent()
    }
    pub fn binding(&self) -> &Binding {
        self.original.binding()
    }
    pub fn bounds(&self) -> &Bounds {
        self.original.bounds()
    }
    pub fn prediction_limit(&self) -> usize {
        PREDICTION_LIMIT
    }
    /// Support and availability are different facts. Neither certifies a label.
    pub fn route(&self, prefix: &[i64]) -> Result<Route> {
        if prefix.is_empty()
            || prefix.len() > PREDICTION_LIMIT
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= self.candidate().vocabulary)
        {
            return Err("prefix outside the declared candidate prediction domain".into());
        }
        let candidate_supported = self
            .candidate()
            .known(&prefix.iter().map(|&t| t as usize).collect::<Vec<_>>());
        let parent_available = prefix.len() <= self.parent().prediction_limit();
        let source = if candidate_supported {
            Source::CandidateSupported
        } else if parent_available {
            Source::Parent
        } else {
            Source::CandidateParentUnavailable
        };
        Ok(Route {
            candidate_supported,
            parent_available,
            source,
        })
    }
    pub fn known(&self, prefix: &[i64]) -> Result<bool> {
        Ok(self.route(prefix)?.candidate_supported)
    }
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        match self.route(prefix)?.source {
            Source::Parent => self.parent().logits(prefix),
            Source::CandidateSupported | Source::CandidateParentUnavailable => Ok(self
                .candidate()
                .logits_tokens(&prefix.iter().map(|&t| t as usize).collect::<Vec<_>>())),
        }
    }
    /// CM41 preserves the actual normalized objective on original supported
    /// record IDs, including repetitions and shorter final minibatches.
    pub fn batch_gradient(&self, input: &Input, ids: &[usize]) -> Result<(f64, Vec<f64>)> {
        self.original.batch_gradient(input, ids)
    }
}
impl causal::metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.candidate().vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.logits(prefix)
            .expect("validated candidate prediction domain")
    }
}

#[cfg(test)]
mod tests;
