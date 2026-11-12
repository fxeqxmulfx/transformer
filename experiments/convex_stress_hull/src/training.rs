//! Imported learners; the new runner removes held-out-based stopping in counts.
use crate::{
    Expert, Result, counts, data, encoded, features, ideal_ce, meta, metrics, passes,
    random::Random, scale, spectral, teacher_ce,
};
use serde_json::{Value, json};
use std::{fs::File, io::Write, time::Instant};

pub struct Outcome {
    pub model: Option<Expert>,
    pub state: Option<Vec<f64>>,
    pub report: Value,
}
pub struct Monitor<'a> {
    pub task: &'a data::Task,
    pub file: File,
    pub started: Instant,
    pub first99: Option<usize>,
}
impl Monitor<'_> {
    fn observe(
        &mut self,
        model: &impl metrics::Predictor,
        update: usize,
        ce: f64,
        extra: Value,
    ) -> Result<()> {
        let primary = metrics::evaluate(model, &self.task.splits[&self.task.selection])?;
        if self.first99.is_none() && passes(&primary) {
            self.first99 = Some(update);
        }
        let ideal = ideal_ce(self.task.vocab, scale(self.task));
        if ce < ideal - 1e-9 {
            return Err("observed CE below common lower bound".into());
        }
        let record = json!({"update":update,"seconds":self.started.elapsed().as_secs_f64(),
            "primary":primary,"training_ce":ce,"ideal_ce_excess":ce-ideal,"diagnostics":extra});
        writeln!(self.file, "{record}")?;
        self.file.flush()?;
        println!(
            "update={update} primary={} ce_excess={:.6e}",
            primary["sequence_accuracy"],
            ce - ideal
        );
        Ok(())
    }
}

pub fn train_moments(
    arm: &str,
    candidate: &str,
    row: &Value,
    p: &Value,
    monitor: &mut Monitor<'_>,
) -> Result<Outcome> {
    use spectral::solver::{Mode, Solver};
    let task = monitor.task;
    let (r, b) = crate::budgets(candidate, task.vocab)?;
    let initial = spectral::Model::new(task.vocab, r, b)?;
    let samples = spectral::samples(&initial, &task.splits["train"])?;
    let count = samples.len();
    let mode = Mode::parse(arm)?;
    let mut solver = Solver::new(
        &initial,
        samples,
        mode,
        p["frequency_workers"].as_u64().unwrap() as usize,
    )?;
    let batch = row["case"]["batch"].as_u64().unwrap() as usize;
    let per_epoch = count.div_ceil(batch);
    let cadence = (row["case"]["observe"].as_u64().unwrap() as usize).min(per_epoch);
    let limit = meta::limit(p, arm, candidate);
    let tolerance = meta::number(p, "ideal_ce_tolerance")?;
    let mut order: Vec<_> = (0..count).collect();
    let mut rng = Random::new(p["seed"].as_u64().unwrap() as u32 + 10_000);
    let (mut updates, mut presented) = (0, 0);
    let (model, alpha, norms, reason) = loop {
        if updates % cadence == 0 || updates == limit {
            let (model, alpha, norms) = solver.snapshot()?;
            let ce = teacher_ce(&model, &task.splits["train"])?;
            monitor.observe(
                &model,
                updates,
                ce,
                json!({"uniform_mixture_alpha":alpha,
                "last_split_residual":solver.residual}),
            )?;
            if updates > 0 && ce - ideal_ce(task.vocab, 1.) <= tolerance {
                break (model, alpha, norms, "ideal_ce_excess");
            }
            if updates == limit {
                break (model, alpha, norms, "budget");
            }
        }
        let within = updates % per_epoch;
        if mode == Mode::Mini && within == 0 {
            rng.shuffle(&mut order);
        }
        let selected = if mode == Mode::Mini {
            &order[within * batch..((within + 1) * batch).min(count)]
        } else {
            &order
        };
        solver.step(selected)?;
        if mode != Mode::Forward || updates == 0 {
            presented += selected.len();
        }
        updates += 1;
    };
    let residual = solver.full_residual()?;
    let report = json!({"status":"complete","stop_reason":reason,"updates":updates,
        "examples_presented":presented,"prefixes_presented":presented,"train_prefixes":count,
        "batch_size":batch,"updates_per_epoch":per_epoch,"cadence":cadence,
        "uniform_mixture_alpha":alpha,"shadow_moment_norms":norms,"fit_defect":solver.fit_defect(&model),
        "full_operator_residual":residual,"last_split_residual":solver.residual,
        "training_gradient_evaluations":if mode==Mode::Forward {0}else{presented}});
    Ok(Outcome {
        model: Some(Expert::Moments(model)),
        state: Some(solver.diagnostic_state().to_vec()),
        report,
    })
}

pub fn train_counts(
    arm: &str,
    row: &Value,
    p: &Value,
    monitor: &mut Monitor<'_>,
) -> Result<Outcome> {
    let task = monitor.task;
    let a = features(task);
    let scale = scale(task);
    let (records, indices) = encoded(a, &task.splits["train"]);
    if records.is_empty() {
        return Err("no training prefixes".into());
    }
    let batch = row["case"]["batch"].as_u64().unwrap() as usize;
    let per_epoch = indices.len().div_ceil(batch);
    let cadence = if arm == "full_batch" {
        1
    } else {
        (row["case"]["observe"].as_u64().unwrap() as usize).min(per_epoch)
    };
    let mut order: Vec<_> = (0..indices.len()).collect();
    let mut rng = Random::new(p["seed"].as_u64().unwrap() as u32 + 10_000);
    let mut report = json!({"train_prefixes":records.len(),"batch_size":batch,
        "updates_per_epoch":per_epoch,"cadence":cadence});
    if arm == "forward_only" {
        rng.shuffle(&mut order);
        let mut trainer = counts::Forward::new(a, scale);
        for chunk in order.chunks(batch) {
            trainer.update_batch(
                &chunk
                    .iter()
                    .flat_map(|&i| &indices[i])
                    .map(|&i| &records[i])
                    .collect::<Vec<_>>(),
            );
        }
        report["updates"] = trainer.updates.into();
        report["examples_presented"] = indices.len().into();
        report["prefixes_presented"] = trainer.observations.into();
        report["training_gradient_evaluations"] = 0.into();
        let model = match trainer.snapshot() {
            Ok(model) => {
                report["status"] = "complete".into();
                report["stop_reason"] = "forward_complete".into();
                let ce = teacher_ce(&model, &task.splits["train"])?;
                monitor.observe(&model, trainer.updates, ce, Value::Null)?;
                Some(Expert::Counts(model))
            }
            Err(failed) => {
                report["status"] = "fit_failure".into();
                report["stop_reason"] = "no_consistent_key".into();
                report["failed_queries"] = json!(failed);
                None
            }
        };
        return Ok(Outcome {
            model,
            state: None,
            report,
        });
    }
    let limit = meta::limit(p, arm, "counts");
    let mut trainer = counts::AdaptiveTrainer::new(a, scale, records.len());
    let (mut presented, mut epochs) = (0, 0);
    let ce = teacher_ce(&trainer, &task.splits["train"])?;
    monitor.observe(&trainer, 0, ce, Value::Null)?;
    let reason = 'training: loop {
        // Preserve counts-v1's fresh order before each shuffle, including full batch.
        for (i, slot) in order.iter_mut().enumerate() {
            *slot = i;
        }
        rng.shuffle(&mut order);
        let size = if arm == "full_batch" {
            order.len()
        } else {
            batch
        };
        for chunk in order.chunks(size) {
            trainer.update_batch(
                &chunk
                    .iter()
                    .flat_map(|&i| &indices[i])
                    .map(|&i| (i, &records[i]))
                    .collect::<Vec<_>>(),
            );
            presented += chunk.len();
            if trainer.updates.is_multiple_of(cadence) || trainer.updates == limit {
                let (ce, gap) = if arm == "full_batch" {
                    let (ce, gap) = trainer.objective(&records);
                    (ce, Some(gap))
                } else {
                    (teacher_ce(&trainer, &task.splits["train"])?, None)
                };
                if gap.is_some_and(|g| g < -1e-8) {
                    return Err("negative full FW gap".into());
                }
                monitor.observe(&trainer, trainer.updates, ce, json!({"fresh_fw_gap":gap}))?;
                if ce - ideal_ce(task.vocab, scale) <= meta::number(p, "ideal_ce_tolerance")? {
                    break 'training "ideal_ce_excess";
                }
                if gap.is_some_and(|g| g <= p["full_gap_tolerance"].as_f64().unwrap()) {
                    break 'training "full_fw_gap";
                }
            }
            if trainer.updates == limit {
                break 'training "budget";
            }
        }
        epochs += 1;
    };
    let (ce, gap) = trainer.objective(&records);
    if gap < -1e-8 {
        return Err("negative endpoint FW gap".into());
    }
    report["status"] = "complete".into();
    report["stop_reason"] = reason.into();
    report["updates"] = trainer.updates.into();
    report["examples_presented"] = presented.into();
    report["prefixes_presented"] = trainer.observations.into();
    report["training_gradient_evaluations"] = trainer.observations.into();
    report["fresh_training_ce"] = ce.into();
    report["fresh_fw_gap"] = gap.into();
    report["unique_prefixes_seen"] = trainer.seen().into();
    report["shared_updates"] = trainer.shared_updates.into();
    report["finished_epochs_before_stop"] = epochs.into();
    Ok(Outcome {
        model: Some(Expert::Counts(trainer.snapshot())),
        state: None,
        report,
    })
}
