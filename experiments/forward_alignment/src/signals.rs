//! Same-state current gradients and isolated tied-embedding contributions.
use crate::{
    config::{Aux, Kernels, Method},
    domain::{
        Result,
        model::{E, W},
    },
    infrastructure::engine::Engine,
    measure::Cost,
};
use serde_json::{Value, json};

pub struct Signals {
    pub ce: Vec<f32>,
    pub pair: Vec<f32>,
    pub auxiliary: Vec<f32>,
    pub local: Vec<f32>,
    pub readout: Vec<f32>,
    pub vocabulary: Vec<f32>,
    pub input: Vec<f32>,
    pub checks: Value,
}
impl Signals {
    /// Immediately after terminal pair backward for the next update batch.
    pub fn new(
        engine: &mut Engine,
        kernels: &Kernels,
        aux: &Aux,
        batch: usize,
        offset: usize,
        terminal: &[Vec<f64>; 2],
        cost: &mut Cost,
    ) -> Result<Self> {
        let ce = terminal[0].iter().map(|&v| v as f32).collect();
        let pair: Vec<f32> = terminal[1].iter().map(|&v| v as f32).collect();
        let input = engine.input_component(batch)?;
        cost.extra_embedding_backwards += 1;
        engine.local_output(kernels, batch)?;
        cost.local_output += 1;
        let mut readout = engine.gradients.read()?;
        readout[E..].fill(0.);
        engine.hidden_objective(aux, kernels, batch, false)?;
        cost.hidden_jacobians += 1;
        engine.vocabulary_objective(aux, kernels)?;
        cost.vocabulary_jacobians += 1;
        let local = engine.gradients.read()?;
        let vocabulary = engine.vocabulary_component(aux, kernels)?;
        cost.vocabulary_jacobians += 1;
        engine.forward(false, offset, batch, true)?;
        cost.forward_batches += 1;
        engine.pilot_gradient(Method::Auxiliary, Some(aux), kernels, batch)?;
        cost.backward_batches += 1;
        cost.hidden_jacobians += 1;
        cost.vocabulary_jacobians += 1;
        cost.extra_embedding_backwards += 1;
        let auxiliary = engine.gradients.read()?;
        let start_b = E + 2 * engine.arm.hidden() * W;
        let same_b = pair[start_b..] == local[start_b..];
        let same_e = (0..E).all(|i| local[i].to_bits() == (readout[i] + vocabulary[i]).to_bits());
        let mut max_error = 0f64;
        let mut violations = 0;
        for i in 0..E {
            let reconstructed = readout[i] as f64 + input[i] as f64;
            let full = pair[i] as f64;
            let error = (full - reconstructed).abs();
            max_error = max_error.max(error);
            violations += usize::from(
                !error.is_finite() || error > 2e-6 + 2e-4 * full.abs().max(reconstructed.abs()),
            );
        }
        let checks = json!({"local_B_equals_pair_B":same_b,"local_E_is_rounded_readout_plus_vocabulary":same_e,
            "input_E_reconstruction_max_error":max_error,"input_E_reconstruction_violations":violations,
            "passed":same_b&&same_e&&violations==0});
        Ok(Self {
            ce,
            pair,
            auxiliary,
            local,
            readout,
            vocabulary,
            input,
            checks,
        })
    }
    pub const RULES: [&'static str; 7] = [
        "ce-bp",
        "pair-bp",
        "auxiliary-bp",
        "local-pair",
        "local-no-vocabulary",
        "local-no-readout",
        "zero-current-gradient",
    ];
    pub fn gradient(&self, index: usize) -> Vec<f32> {
        match index {
            0 => self.ce.clone(),
            1 => self.pair.clone(),
            2 => self.auxiliary.clone(),
            3 => self.local.clone(),
            4 | 5 => {
                let mut g = self.local.clone();
                g[..E].copy_from_slice(if index == 4 {
                    &self.readout[..E]
                } else {
                    &self.vocabulary[..E]
                });
                g
            }
            _ => vec![0.; self.local.len()],
        }
    }
}
