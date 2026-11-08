//! One full-CE trajectory. Canonical export/hash and state validation are
//! charged to scheduled evaluation; report/checkpoint I/O is excluded.
use crate::{
    application::{Backend, Report},
    domain::{
        Result,
        data::{Data, Plan},
        model::{self, Arm},
    },
    infrastructure::storage,
    inputs::require,
    state,
    trainer::Trainer,
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
pub struct Worker {
    pub trainer: Trainer,
    pub step: usize,
    pub status: String,
    pub history: Vec<Value>,
    chunks: Vec<Value>,
    setup: f64,
    train: f64,
    evaluation: f64,
    error: Option<String>,
}
impl Worker {
    pub fn new(
        arm: Arm,
        seed: u64,
        compact: bool,
        data: &Data,
        plan: &Plan,
        shared: f64,
        cuda: &Cuda,
    ) -> Result<Self> {
        cuda.sync()?;
        let started = Instant::now();
        let trainer = Trainer::new(
            arm,
            &model::initialize(arm, seed),
            data,
            plan,
            compact,
            cuda,
        )?;
        cuda.sync()?;
        Ok(Self {
            trainer,
            step: 0,
            status: "running".into(),
            history: vec![],
            chunks: vec![],
            setup: shared + started.elapsed().as_secs_f64(),
            train: 0.,
            evaluation: 0.,
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
        let started = Instant::now();
        let result = (|| -> Result<Value> {
            let r = Report {
                step: self.step,
                train: self.trainer.evaluate(false)?,
                validation: self.trainer.evaluate(true)?,
            };
            let canonical = self.trainer.state()?;
            require(
                state::valid(self.trainer.arm(), &canonical),
                "invalid scheduled state",
            )?;
            let mut row = storage::report_value(&r);
            row["state_sha256"] =
                json!(storage::hash(&storage::state_bytes(self.step, &canonical)));
            row["state_valid"] = json!(true);
            cuda.sync()?;
            Ok(row)
        })();
        self.evaluation += started.elapsed().as_secs_f64();
        let mut row = result?;
        row["active_seconds"] = json!(self.active());
        row["training_seconds"] = json!(self.train);
        self.history.push(row);
        Ok(())
    }
    pub fn chunk(&mut self, end: usize, plan: &Plan, cuda: &Cuda) -> Result<()> {
        cuda.sync()?;
        let started = Instant::now();
        let from = self.step;
        let result = (|| -> Result<()> {
            for step in from + 1..=end {
                let offset = plan.boundaries[step - 1];
                self.trainer
                    .train_step(offset, plan.boundaries[step] - offset, step)?;
                self.step = step;
            }
            cuda.sync()?;
            Ok(())
        })();
        let seconds = started.elapsed().as_secs_f64();
        self.train += seconds;
        self.chunks
            .push(json!({"from":from,"to":self.step,"seconds":seconds,"passed":result.is_ok()}));
        result
    }
    pub fn report(&self) -> Value {
        json!({"method":self.trainer.name(),"status":self.status,"completed_steps":self.step,"error":self.error,
            "control_verified":true,"setup_seconds":self.setup,"training_seconds":self.train,"evaluation_seconds":self.evaluation,"active_seconds":self.active(),
            "training_seconds_per_update":if self.step>0{Some(self.train/self.step as f64)}else{None},
            "allocated_payload_bytes":self.trainer.bytes(),"moment_bytes":self.trainer.moment_bytes(),
            "transient_source_moments_during_setup":self.trainer.name()=="compact-adamw"&&self.trainer.arm()!=Arm::Swiglu,
            "history":self.history,"chunks":self.chunks,
            "cost":{"training_forward_batches":self.step,"full_backwards":self.step,"optimizer_kernels":self.step,"evaluation_forward_batches":20*self.history.len(),"canonical_exports":self.history.len()}})
    }
    pub fn save(&self, directory: &Path) -> Result<Value> {
        let state = self.trainer.state()?;
        let bytes = storage::state_bytes(self.step, &state);
        fs::write(
            directory.join(format!("{}.bin", self.trainer.name())),
            &bytes,
        )?;
        Ok(
            json!({"sha256":storage::hash(&bytes),"valid":state::valid(self.trainer.arm(),&state),"step":self.step}),
        )
    }
}
