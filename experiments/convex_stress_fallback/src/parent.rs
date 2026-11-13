//! Dispatch by saved model representation. No benchmark name or stored
//! evaluation prediction enters the forward pass. Other families are refused.
use crate::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "model", deny_unknown_fields)]
pub enum Parent {
    Causal(causal::model::Model),
    Counts(counts::CountParent),
}

pub fn causal_bounds(model: &causal::model::Model) -> Result<Bounds> {
    model.validate()?;
    let states: usize = model.blocks.iter().map(|b| b.states()).sum();
    let entries: usize = model.blocks.iter().map(|b| b.code.len()).sum();
    Ok(Bounds {
        real_slots: model.coefficients.len() as u64,
        stored_scalars: (model.coefficients.len() + 2 * entries) as u64,
        radius: model.radius,
        buffer_bytes: causal::model::memory_bound(model.vocabulary, states, entries)?,
    })
}

impl Parent {
    /// Accept the original typed Counts checkpoint or an explicit Causal
    /// checkpoint. Reject other saved kinds before any prediction is attempted.
    pub fn from_checkpoint(value: serde_json::Value) -> Result<Self> {
        let parent: Self = serde_json::from_value(value)?;
        parent.bounds()?;
        Ok(parent)
    }

    pub fn from_saved(kind: &str, value: serde_json::Value) -> Result<Self> {
        let parent = match kind {
            "causal_pairs" => Self::Causal(serde_json::from_value(value)?),
            "counts" => Self::Counts(counts::CountParent::from_saved(value)?),
            _ => return Err("unsupported saved parent representation".into()),
        };
        parent.bounds()?;
        Ok(parent)
    }
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Causal(m) => (m.vocabulary, m.context),
            Self::Counts(m) => (m.features.base.vocab, m.features.base.context),
        }
    }
    pub fn bounds(&self) -> Result<Bounds> {
        match self {
            Self::Causal(m) => causal_bounds(m),
            Self::Counts(m) => {
                let bounds = m.validate()?;
                if m.column_l1()?
                    .iter()
                    .any(|&n| !n.is_finite() || n > bounds.radius)
                {
                    return Err("actual count-parent columns exceed the charged radius".into());
                }
                Ok(bounds)
            }
        }
    }
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        let (v, context) = self.shape();
        check_prefix(prefix, v, context)?;
        match self {
            Self::Causal(m) => {
                Ok(m.logits_tokens(&prefix.iter().map(|&x| x as usize).collect::<Vec<_>>()))
            }
            Self::Counts(m) => Ok(m
                .complex_logits(prefix)?
                .into_iter()
                .map(|z| z.re)
                .collect()),
        }
    }
}
