//! Replay the ordinary sum trajectory without repeating any gradient work.
//! Only the 31 staircase labels choose a model. The original hidden split
//! independently verifies the selected model after selection.
use super::{Split, new_engine, observed_phase_labels};
use crate::{
    Result, checkpoint_proposal,
    engine::adamw::FIRST_NOISE_STDDEV,
    hash,
    random::Random,
    suite::{load_weights, publish, save_weights},
    sum8_spectrum::staircase_signs,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

pub(crate) fn checkpoint_proposals(
    cell: &Path,
    seed: u32,
    out: &Path,
    first: usize,
) -> Result<Value> {
    let bytes = fs::read(cell.join("result.json"))?;
    let record: Value = serde_json::from_slice(&bytes)?;
    let native_seed = u32::try_from(record["seeds"][0].as_u64().ok_or("missing model seed")?)?;
    let end = usize::try_from(
        record["optimizer"]["step"]
            .as_u64()
            .ok_or("missing final step")?,
    )?;
    if record["protocol"] != "argand-sum8-adamw-wsd-v1" || first > end {
        return Err(
            "expected an ordinary AdamW sum trajectory and a proposal within its cap".into(),
        );
    }
    let identity = json!({"protocol":"argand-sum8-checkpoint-proposals-v1","source_sha256":hash(&bytes),
        "native_seed":native_seed,"proposal_seed":seed,"first_eligible_update":first,"stddev":FIRST_NOISE_STDDEV,
        "selection":"strict hidden margins reconstructed from the 31 observed staircase signs"});
    fs::create_dir_all(out)?;
    if fs::canonicalize(out)? == fs::canonicalize(cell)? {
        return Err("proposal output must be separate from the source checkpoint".into());
    }
    let lock = fs::File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(out.join("cell.lock"))?;
    lock.lock()?;
    let mut model = new_engine(native_seed)?;
    if out.join("result.json").try_exists()? {
        let result: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
        if result["identity"] != identity {
            return Err("incompatible sum proposal report".into());
        }
        load_weights(out, "weights.f32", &result["weights_sha256"], &mut model)?;
        if Split::new(false).score(&mut model)? != result["hidden"] {
            return Err("saved sum proposal failed replay".into());
        }
        return Ok(result);
    }
    let (diag, edge) = observed_phase_labels();
    let signs = staircase_signs(&diag, &edge);
    let certified = Split::with_labels(false, |w| u32::from(signs[w as usize] < 0.0));
    let mut rng = Random::new(seed);
    let mut points = vec![(0, &record["initial"])];
    let mut previous = 0;
    for point in record["milestones"]
        .as_array()
        .ok_or("missing ordinary checkpoints")?
    {
        let step = usize::try_from(point["step"].as_u64().ok_or("missing milestone step")?)?;
        if step <= previous || step >= end {
            return Err("invalid ordinary checkpoint order".into());
        }
        points.push((step, point));
        previous = step;
    }
    points.push((end, &record));
    let mut checks = Vec::new();
    for (step, point) in points {
        if step != 0 {
            let file = if step == end {
                "weights.f32".to_owned()
            } else {
                format!("weights-{step}.f32")
            };
            load_weights(cell, &file, &point["weights_sha256"], &mut model)?;
        }
        let decision = if step < first {
            let ordinary = certified.score(&mut model)?;
            json!({"accepted":ordinary["all_strict"],"selected":"ordinary","ordinary":ordinary,"proposal":null})
        } else {
            checkpoint_proposal::evaluate(&mut model, &mut rng, FIRST_NOISE_STDDEV, |m| {
                certified.score(m)
            })?
        };
        if decision["ordinary"] != point["hidden"] {
            return Err(
                "observed-only reconstruction disagrees with the ordinary checkpoint replay".into(),
            );
        }
        let accepted = decision["accepted"].as_bool().ok_or("missing decision")?;
        checks.push(json!({"step":step,"decision":decision}));
        if accepted || step == end {
            let measured = certified.score(&mut model)?;
            let (_, digest) = save_weights(out, "weights.f32", &model)?;
            if step > 0 && decision["selected"] == "ordinary" && point["weights_sha256"] != digest {
                return Err("a rejected proposal changed the checkpoint weights".into());
            }
            let mut replay = new_engine(native_seed)?;
            load_weights(out, "weights.f32", &json!(digest), &mut replay)?;
            let hidden = Split::new(false).score(&mut replay)?;
            if measured != hidden || hidden["all_strict"] != accepted {
                return Err(
                    "independent sum proposal replay disagrees with the certificate".into(),
                );
            }
            let result = json!({"identity":identity,"step":step,"accepted":accepted,"checks":checks,
                "weights_sha256":digest,"hidden":hidden,"observed":Split::new(true).score(&mut replay)?,"replayed":true});
            publish(out, "result.json", &serde_json::to_vec_pretty(&result)?)?;
            return Ok(result);
        }
    }
    Err("missing terminal sum checkpoint".into())
}
