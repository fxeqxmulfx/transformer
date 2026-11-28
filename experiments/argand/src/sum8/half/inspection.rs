//! Inspect saved weights in a separate engine; training never sees held-out labels.
use super::*;
use crate::{engine::adamw::DECAY, suite::read_floats, sum8_spectrum::logit_gaps};

fn weights_at(cell: &Path, step: usize) -> Result<(Vec<f32>, Value)> {
    let mut file = format!("weights-{step}.f32");
    let record = cell.join("result.json");
    let digest = if record.try_exists()? {
        let record: Value = serde_json::from_slice(&fs::read(record)?)?;
        let point = if record["optimizer"]["step"] == step {
            file = "weights.f32".into();
            &record
        } else {
            record["milestones"]
                .as_array()
                .and_then(|points| points.iter().find(|p| p["step"] == step))
                .ok_or("missing saved checkpoint")?
        };
        point["weights_sha256"].clone()
    } else {
        json!(crate::hash(&fs::read(cell.join(&file))?))
    };
    Ok((read_floats(&cell.join(file), &digest)?, digest))
}

fn norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt()
}

fn inspect(model: &mut Engine, weights: &[f32], train: &Split, hidden: &Split) -> Result<Value> {
    model.restore(weights)?;
    let grads = |m: &mut Engine, s: &Split| {
        m.mean_loss_gradient(&s.tokens, &s.labels, s.tokens.len() / LENGTH, LENGTH)
    };
    let train_grad = grads(model, train)?;
    let hidden_grad = grads(model, hidden)?;
    let train_norm = norm(&train_grad);
    let hidden_norm = norm(&hidden_grad);
    let train_score = train.score(model)?;
    let hidden_score = hidden.score(model)?;
    let table = logit_gaps(model)?;
    let train_ids = words();
    let misses: Vec<_> = (0..256)
        .filter(|w| {
            let gap = table[*w as usize];
            train_ids.binary_search(w).is_err() && (if parity(*w) == 0 { gap } else { -gap }) <= 0.
        })
        .collect();
    if misses.len()
        + hidden_score["correct"]
            .as_u64()
            .ok_or("missing hidden count")? as usize
        != 128
    {
        return Err("table and split evaluations disagree".into());
    }
    let mut differences = Vec::new();
    if train_norm > 0.0 {
        for h in [0.001, 0.0003] {
            let mut loss = |signed_h: f64| -> Result<f64> {
                let perturbed: Vec<_> = weights
                    .iter()
                    .zip(&train_grad)
                    .map(|(&w, &g)| (f64::from(w) + signed_h * f64::from(g) / train_norm) as f32)
                    .collect();
                model.restore(&perturbed)?;
                train.score(model)?["ce"]
                    .as_f64()
                    .ok_or_else(|| "invalid train CE".into())
            };
            let slope = (loss(h)? - loss(-h)?) / (2. * h);
            differences.push(
                json!({"h":h,"central_slope":slope,"relative_error":(slope/train_norm-1.0).abs()}),
            );
        }
    }
    Ok(
        json!({"train":train_score,"hidden":hidden_score,"wrong_hidden_ids":misses,
        "train_gradient_norm":train_norm,"nonzero_train_gradient_coordinates":train_grad.iter().filter(|&&g|g!=0.).count(),
        "hidden_gradient_norm":hidden_norm,"finite_difference_along_unit_train_gradient":differences}),
    )
}

pub(crate) fn inspect_checkpoints(
    dir: &Path,
    draw: usize,
    first: usize,
    last: usize,
) -> Result<Value> {
    let (budget, plan) = manifest(dir, None, true, Method::AdamWHalf)?;
    if draw >= RUNS || first == 0 || first >= last || last > budget {
        return Err("expected two saved steps in increasing order and a registered draw".into());
    }
    let cell = dir.join(format!("draw-{draw}"));
    let (a, da) = weights_at(&cell, first)?;
    let (b, db) = weights_at(&cell, last)?;
    if a.len() != b.len() {
        return Err("weight lengths differ".into());
    }
    let mut model = engine_with_batch(read_seed(&plan["seeds"][draw], 0)?, 128)?;
    let (train, hidden) = (split(true), split(false));
    let wa = inspect(&mut model, &a, &train, &hidden)?;
    let wb = inspect(&mut model, &b, &train, &hidden)?;
    let scale = (first..last)
        .map(|i| f64::from(1.0_f32 - RatePlan::Wsd.rate(wsd(budget), i) * DECAY))
        .product::<f64>();
    let distance = |scale: f64| {
        a.iter()
            .zip(&b)
            .map(|(&x, &y)| (f64::from(y) - scale * f64::from(x)).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let changed = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let initial_norm = norm(&a);
    Ok(json!({
        "draw":draw, "seeds":plan["seeds"][draw], "first_step":first, "last_step":last,
        "first_sha256":da, "last_sha256":db, "parameters":a.len(), "changed_coordinates":changed,
        "relative_displacement":distance(1.)/initial_norm, "pure_decay_scale":scale,
        "relative_distance_from_pure_decay":distance(scale)/initial_norm, "first":wa, "last":wb,
    }))
}
