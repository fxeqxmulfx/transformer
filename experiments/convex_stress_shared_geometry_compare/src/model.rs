//! CM38 actual complete coordinate blocks, fixed training coverage and old route.
use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Saved {
    pub parent_sha256: String,
    pub prior: fresh::training::Outcome,
    pub supplement: Option<fresh::training::Outcome>,
}
pub(crate) fn coverage(input: &fresh::Input) -> Result<Vec<bool>> {
    input.validate()?;
    let mut present = vec![false; input.blocks];
    for r in &input.records {
        present[input.route[r.cell]] = true;
    }
    Ok(present)
}
pub(crate) fn use_supplement(prior: bool, supplement: Option<bool>) -> bool {
    !prior && supplement == Some(true)
}
pub(crate) struct Role {
    pub old: old::Model,
    pub outcome: fresh::training::Outcome,
    pub covered: Vec<bool>,
}
impl Role {
    fn load(value: Value, outcome: fresh::training::Outcome, sha: &str) -> Result<Self> {
        let binding: old::Binding = serde_json::from_value(value["binding"].clone())?;
        let old = old::Model::load(value, &binding)?;
        let input = fresh::Input::from_source(old.source(), old.records().to_vec(), sha.into())?;
        let same_mode = matches!(
            (old.endpoint().mode, outcome.policy.mode),
            (old::Mode::Full, fresh::Mode::Full)
                | (old::Mode::Mini, fresh::Mode::Mini)
                | (old::Mode::Forward, fresh::Mode::Forward)
        );
        if input != outcome.model.input || !same_mode {
            return Err("saved candidate route/records differ from actual old model".into());
        }
        fresh::training::replay(&input, &outcome)?;
        outcome.model.validate()?;
        let covered = coverage(&input)?;
        Ok(Self {
            old,
            outcome,
            covered,
        })
    }
    pub fn covered(&self, cell: usize) -> bool {
        self.covered[self.outcome.model.input.route[cell]]
    }
    fn logits(&self, cell: usize) -> Result<Vec<f64>> {
        if self.covered(cell) {
            self.outcome.model.logits_cell(cell)
        } else {
            self.old.logits_cell(cell)
        }
    }
    fn independent(&self, cell: usize) -> Result<Vec<f64>> {
        if !self.covered(cell) {
            return self.old.logits_cell(cell);
        }
        self.outcome.model.independent_logits(cell)
    }
    fn info(&self) -> Value {
        json!({"input_sha256":self.outcome.input_sha256,"initial_model_sha256":self.outcome.initial_model_sha256,
            "model_sha256":fresh::digest(&self.outcome.model).expect("serializable head"),"policy":self.outcome.policy,
            "initialization":{"complete_state_in_saved_outcome":true,"sha256":fresh::digest(&self.outcome.initialization).expect("serializable constructor"),
              "selected":self.outcome.initialization.selected,"bank":self.outcome.model.bank,"ridge":self.outcome.initialization.refit.ridge,
              "direction":self.outcome.initialization.refit.direction,"calibrated":self.outcome.initialization.refit.calibrated,
              "gain":self.outcome.initialization.gain,"conditional_fold_score":self.outcome.initialization.pooled_scores[self.outcome.initialization.selected],
              "candidate_states":self.outcome.initialization.candidates.len(),"ce_gradients":self.outcome.initialization.ce_gradients},
            "initial_ce":self.outcome.initial_ce,"final_ce":self.outcome.final_ce,
            "work":self.outcome.work,"new_bounds":self.outcome.bounds,"old_bounds":self.old.bounds(),
            "covered_blocks":self.covered.iter().filter(|&&x|x).count(),"total_blocks":self.covered.len(),
            "combined_radius_bound":fresh::RADIUS+self.old.source().radius,"all_original_training_records_covered":true})
    }
}
pub(crate) struct Model {
    pub prior: Role,
    pub supplement: Option<Role>,
}
impl Model {
    pub fn load(mut parent: Value, saved: Saved) -> Result<Self> {
        let prior = Role::load(parent["prior"].take(), saved.prior, &saved.parent_sha256)?;
        let supplement = match (saved.supplement, parent.get_mut("supplement")) {
            (Some(outcome), Some(value)) if !value.is_null() => {
                Some(Role::load(value.take(), outcome, &saved.parent_sha256)?)
            }
            (None, None) => None,
            (None, Some(value)) if value.is_null() => None,
            _ => return Err("saved role presence differs".into()),
        };
        Ok(Self { prior, supplement })
    }
    pub fn routed(&self, cell: usize) -> (&Role, &'static str) {
        if use_supplement(
            self.prior.old.source().known(cell),
            self.supplement.as_ref().map(|r| r.old.source().known(cell)),
        ) {
            (self.supplement.as_ref().unwrap(), "supplement")
        } else {
            (&self.prior, "prior")
        }
    }
    pub fn cell(&self, prefix: &[i64]) -> usize {
        prefix[prefix.len() - 2] as usize * self.prior.outcome.model.input.n
            + prefix[prefix.len() - 1] as usize
    }
    pub fn old_logits(&self, cell: usize) -> Result<Vec<f64>> {
        self.routed(cell).0.old.logits_cell(cell)
    }
    pub fn logits(&self, cell: usize) -> Result<Vec<f64>> {
        self.routed(cell).0.logits(cell)
    }
    pub fn independent(&self, cell: usize) -> Result<Vec<f64>> {
        self.routed(cell).0.independent(cell)
    }
    pub fn info(&self) -> Value {
        json!({"prior":self.prior.info(),"supplement":self.supplement.as_ref().map(Role::info),
            "training_objective":"sum of independent role CE objectives; composite observed CE separately evaluated",
            "new_optimizer_updates":std::iter::once(&self.prior).chain(self.supplement.iter()).map(|r|r.outcome.work.gradient_batches).sum::<usize>(),
            "new_gradient_batches":std::iter::once(&self.prior).chain(self.supplement.iter()).map(|r|r.outcome.work.gradient_batches).sum::<usize>(),
            "complete_real_coefficient_slots":std::iter::once(&self.prior).chain(self.supplement.iter()).map(|r|r.outcome.model.input.blocks*(1+2*r.outcome.model.bank.states())*r.outcome.model.input.vocabulary+r.old.source().coefficients.len()).sum::<usize>(),
            "stored_new_shared_parameters":std::iter::once(&self.prior).chain(self.supplement.iter()).map(|r|r.outcome.model.theta.len()).sum::<usize>(),
            "complete_combined_radius_bound":std::iter::once(&self.prior).chain(self.supplement.iter()).map(|r|fresh::RADIUS+r.old.source().radius).sum::<f64>(),
            "new_local_row_updates":0,"unconditional_identification":false})
    }
}
pub(crate) struct Predictor<'a> {
    pub model: &'a Model,
    pub parent: bool,
    pub calls: std::cell::Cell<u64>,
}
impl metrics::Predictor for Predictor<'_> {
    fn vocab(&self) -> usize {
        self.model.prior.outcome.model.input.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.calls.set(self.calls.get() + 1);
        let cell = self.model.cell(prefix);
        if self.parent {
            self.model.old_logits(cell)
        } else {
            self.model.logits(cell)
        }
        .expect("prevalidated original input")
    }
}
pub(crate) fn argmax(z: &[f64]) -> Result<usize> {
    if z.is_empty() || z.iter().any(|v| !v.is_finite()) {
        return Err("invalid actual logits".into());
    }
    Ok((0..z.len())
        .max_by(|&a, &b| z[a].total_cmp(&z[b]).then(b.cmp(&a)))
        .unwrap())
}
