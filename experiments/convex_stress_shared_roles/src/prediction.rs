//! Bound inference for CM40's actual complex endpoint. The original token
//! domain and the last-two-token view are explicit; earlier tokens cannot
//! alter the view. Full-basis support and exact observed inputs stay distinct.
//! Ported kernel/model.rs at ec5f839d: row is last, column is penultimate.
use crate::*;
pub const MAX_PREFIX: usize = 4096;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Support {
    Supported,
    UnmatchedActive,
    UnknownAxis,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub logits: Vec<f64>,
    pub support: Support,
    pub exact_observed: bool,
}
/// A verified immutable borrow prevents mutation after endpoint binding.
pub struct CheckedHead<'a> {
    model: &'a Model,
}
impl<'a> CheckedHead<'a> {
    pub fn bind(
        discovery: &discovery::Report,
        records: &[records::Record],
        policies: &[training::Policy; 3],
        report: &'a binding::Report,
        mode: training::Mode,
    ) -> Result<Self> {
        binding::check(discovery, records, policies, report)?;
        let index = match mode {
            training::Mode::Full => 0,
            training::Mode::Mini => 1,
            training::Mode::Forward => 2,
        };
        let out = report.endpoints[index]
            .training
            .as_ref()
            .ok_or("requested mode was refused")?;
        Ok(Self { model: &out.model })
    }
    pub fn predict(&self, prefix: &[usize]) -> Result<Prediction> {
        if !(2..=MAX_PREFIX).contains(&prefix.len()) || prefix.iter().any(|&x| x >= self.model.n) {
            return Err("prefix is outside the declared shared-role input domain".into());
        }
        let cell = prefix[prefix.len() - 1] + self.model.n * prefix[prefix.len() - 2];
        let location = self.model.location(cell);
        let support = match self.model.latent(cell) {
            None => Support::UnknownAxis,
            Some(_) if self.model.known(cell) => Support::Supported,
            Some(_) => Support::UnmatchedActive,
        };
        Ok(Prediction {
            logits: self.model.logits_state(location.0, location.1),
            support,
            exact_observed: self.model.blocks[location.0]
                .observed
                .binary_search(&cell)
                .is_ok(),
        })
    }
}
