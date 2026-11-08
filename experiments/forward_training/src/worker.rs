//! One timed training trajectory; reporting and checkpoint I/O are excluded.
use crate::{
    application::{Backend, Report},
    config::{Aux, Kernels, Method, Prototypes},
    domain::{
        Result,
        data::{Data, Plan},
        model::{self, Arm},
    },
    infrastructure::{engine::Engine, storage},
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};

pub struct Worker {
    pub engine: Engine,
    pub method: Method,
    pub aux: Option<Aux>,
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
        method: Method,
        data: &Data,
        plan: &Plan,
        shared_seconds: f64,
        cuda: &Cuda,
    ) -> Result<Self> {
        cuda.sync()?;
        let start = Instant::now();
        let engine = Engine::new(arm, &model::initialize(arm, seed), data, plan)?;
        let p = method.auxiliary().then(|| Prototypes::new(arm, seed));
        let aux = p
            .as_ref()
            .map(|p| Aux::new(cuda, arm, p, method))
            .transpose()?;
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
    pub fn evaluate(&mut self, cuda: &Cuda, original: &[Value]) -> Result<()> {
        cuda.sync()?;
        let start = Instant::now();
        let result = (|| -> Result<_> {
            let r = Report {
                step: self.step,
                train: self.engine.evaluate(false)?,
                validation: self.engine.evaluate(true)?,
            };
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
        if self.method == Method::Ce {
            let source = &original[self.step / 100];
            let matched = [("train", &report.train), ("validation", &report.validation)]
                .iter()
                .all(|(key, m)| {
                    source["step"] == self.step
                        && source[*key]["correct"] == m.correct
                        && source[*key]["count"] == m.count
                        && (source[*key]["loss"].as_f64().unwrap() - m.loss).abs() <= 1e-12
                });
            row["source_replay_match"] = json!(matched);
        }
        self.history.push(row);
        Ok(())
    }
    pub fn chunk(&mut self, end: usize, plan: &Plan, cuda: &Cuda, kernels: &Kernels) -> Result<()> {
        cuda.sync()?;
        let start = Instant::now();
        let from = self.step;
        let result = (|| -> Result<()> {
            for step in from + 1..=end {
                let offset = plan.boundaries[step - 1];
                self.engine.pilot_step(
                    self.method,
                    self.aux.as_ref(),
                    kernels,
                    offset,
                    plan.boundaries[step] - offset,
                    step,
                )?;
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
        json!({"method":self.method.name(),"status":self.status,"completed_steps":self.step,
            "error":self.error,"setup_seconds":self.setup,"training_seconds":self.train,
            "evaluation_seconds":self.evaluation,"active_seconds":self.active(),
            "training_seconds_per_update":if self.step>0 {Some(self.train/self.step as f64)} else {None},
            "allocated_payload_bytes":self.engine.allocated_bytes()+aux_bytes,"auxiliary_payload_bytes":aux_bytes,
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
