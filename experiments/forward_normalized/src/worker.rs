//! Adapted from forward_training/worker.rs at c9995ba5. Changes only rule
//! dispatch, normalized-A scratch, and charged evaluation state checks.
use crate::{
    application::{Backend, Report},
    config::{Aux, Kernels, Prototypes},
    coordinates,
    domain::{
        Result,
        data::{Data, Plan},
        model::{self, Arm},
    },
    infrastructure::{engine::Engine, storage},
    normalized::Normalizer,
    normalized_kernel,
    rule::Rule,
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};

pub struct Worker {
    pub engine: Engine,
    pub method: Rule,
    pub aux: Option<Aux>,
    pub normalizer: Option<Normalizer>,
    stats: Vec<f64>,
    pub step: usize,
    pub status: String,
    pub history: Vec<Value>,
    chunks: Vec<Value>,
    setup: f64,
    train: f64,
    evaluation: f64,
    prototypes: Value,
    error: Option<String>,
}
impl Worker {
    pub fn new(
        arm: Arm,
        seed: u64,
        method: Rule,
        data: &Data,
        plan: &Plan,
        shared_seconds: f64,
        cuda: &Cuda,
    ) -> Result<Self> {
        cuda.sync()?;
        let start = Instant::now();
        let engine = Engine::new(arm, &model::initialize(arm, seed), data, plan)?;
        let p = method
            .method()
            .auxiliary()
            .then(|| Prototypes::new(arm, seed));
        let aux = p
            .as_ref()
            .map(|p| Aux::new(cuda, arm, p, method.method()))
            .transpose()?;
        let normalizer = if method == Rule::Normalized {
            Some(Normalizer::new(cuda, arm)?)
        } else {
            None
        };
        cuda.sync()?;
        let setup = shared_seconds + start.elapsed().as_secs_f64();
        // Provenance serialization is outside setup timing.
        let prototypes = p
            .map(|p| {
                json!({
            "vocabulary_sha256":storage::hash(&storage::floats(&p.vocabulary)),
            "hidden_sha256":storage::hash(&storage::floats(&p.hidden))})
            })
            .unwrap_or(Value::Null);
        Ok(Self {
            engine,
            method,
            aux,
            normalizer,
            stats: vec![],
            step: 0,
            status: "running".into(),
            history: vec![],
            chunks: vec![],
            setup,
            train: 0.,
            evaluation: 0.,
            prototypes,
            error: None,
        })
    }
    pub fn active(&self) -> f64 {
        self.setup + self.train + self.evaluation
    }
    pub fn fail(&mut self, error: impl ToString) {
        self.status = "error".into();
        self.error = Some(error.to_string());
    }
    pub fn evaluate(&mut self, cuda: &Cuda) -> Result<()> {
        cuda.sync()?;
        let start = Instant::now();
        let result = (|| -> Result<_> {
            let r = Report {
                step: self.step,
                train: self.engine.evaluate(false)?,
                validation: self.engine.evaluate(true)?,
            };
            let state = self.engine.state()?;
            crate::inputs::require(
                crate::config::valid_state(self.engine.arm, &state),
                "invalid scheduled state/ties",
            )?;
            if let Some(normalizer) = &self.normalizer {
                self.stats = normalizer.stats.read()?;
                crate::inputs::require(
                    self.stats.iter().all(|v| v.is_finite()),
                    "nonfinite norm statistics",
                )?;
                let a = coordinates::expanded_groups(self.engine.arm)[1].1.clone();
                crate::inputs::require(
                    state.first[a.clone()]
                        .iter()
                        .chain(&state.second[a])
                        .all(|&v| v == 0.),
                    "A moments changed",
                )?;
            }
            cuda.sync()?;
            Ok(r)
        })();
        let seconds = start.elapsed().as_secs_f64();
        self.evaluation += seconds;
        let report = result?;
        let mut row = storage::report_value(&report);
        row["evaluation_seconds"] = json!(seconds);
        row["active_seconds"] = json!(self.active());
        row["training_seconds"] = json!(self.train);
        row["state_valid"] = json!(true);
        row["normalization_stats"] = json!(self.stats);
        self.history.push(row);
        Ok(())
    }
    pub fn chunk(
        &mut self,
        end: usize,
        plan: &Plan,
        cuda: &Cuda,
        kernels: &Kernels,
        normalized: &normalized_kernel::kernels::Kernels,
    ) -> Result<()> {
        cuda.sync()?;
        let start = Instant::now();
        let from = self.step;
        let result = (|| -> Result<()> {
            for step in from + 1..=end {
                let offset = plan.boundaries[step - 1];
                let batch = plan.boundaries[step] - offset;
                match self.method {
                    Rule::Ce => self.engine.train_step(offset, batch, step)?,
                    Rule::Local => self.engine.pilot_step(
                        self.method.method(),
                        self.aux.as_ref(),
                        kernels,
                        offset,
                        batch,
                        step,
                    )?,
                    Rule::Normalized => {
                        self.engine.forward(false, offset, batch, true)?;
                        self.engine.pilot_gradient(
                            self.method.method(),
                            self.aux.as_ref(),
                            kernels,
                            batch,
                        )?;
                        self.engine.normalized_update(
                            step,
                            self.normalizer.as_ref().ok_or("missing normalizer")?,
                            normalized,
                        )?;
                    }
                }
                self.step = step;
            }
            cuda.sync()?;
            Ok(())
        })();
        let seconds = start.elapsed().as_secs_f64();
        self.train += seconds;
        self.chunks
            .push(json!({"from":from,"to":self.step,"seconds":seconds,"passed":result.is_ok()}));
        result
    }
    pub fn report(&self) -> Value {
        let aux_bytes = self.aux.as_ref().map_or(0, Aux::bytes);
        let norm_bytes = self.normalizer.as_ref().map_or(0, Normalizer::bytes);
        json!({"method":self.method.name(),"status":self.status,"completed_steps":self.step,
            "error":self.error,"setup_seconds":self.setup,"training_seconds":self.train,
            "evaluation_seconds":self.evaluation,"active_seconds":self.active(),
            "training_seconds_per_update":if self.step>0 {Some(self.train/self.step as f64)} else {None},
            "allocated_payload_bytes":self.engine.allocated_bytes()+aux_bytes+norm_bytes,"auxiliary_payload_bytes":aux_bytes,"norm_payload_bytes":norm_bytes,
            "normalization_stats":self.stats,"control_verified":true,
            "cost":{"training_forward_batches":self.step,"terminal_backwards":if self.method==Rule::Ce {self.step}else{0},
            "each_local_jacobian":if self.method==Rule::Ce {0}else{self.step},
            "norm_kernels":if self.method==Rule::Normalized {2*self.step}else{0},
            "A_update_kernels":if self.method==Rule::Normalized {self.step}else{0},
            "source_adamw_kernels":if self.method==Rule::Normalized {2*self.step}else{self.step},
            "evaluation_forward_batches":20*self.history.len()},
            "prototypes":self.prototypes,"history":self.history,"chunks":self.chunks})
    }
    pub fn save(&self, directory: &Path) -> Result<Value> {
        let state = self.engine.state()?;
        let valid = crate::config::valid_state(self.engine.arm, &state);
        let bytes = storage::state_bytes(self.step, &state);
        fs::write(
            directory.join(format!("{}.bin", self.method.name())),
            &bytes,
        )?;
        Ok(json!({"step":self.step,"sha256":storage::hash(&bytes),"valid":valid}))
    }
}
