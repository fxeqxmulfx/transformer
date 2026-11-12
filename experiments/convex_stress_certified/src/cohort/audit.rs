//! Raw CM16 auditor copied unchanged from a56cf4f0, except this import path.
//! The reflection, the decoding, the Fourier sums, the cross-entropy and the
//! split metrics are recomputed here from the saved numbers; none of them calls
//! the producer's decoder, cosine transform, projector, loss or metric code.
//! The frozen data construction and the LAPACK SVD are shared; all bounds are
//! floating-point checks, not interval or exact-real certificates.
use convex_stress_paired::{C, Model, Result, linalg};
use convex_stress_relabel::data;
use serde_json::{Value, json};

pub fn close(a: &Value, b: &Value, tolerance: f64) -> Result<()> {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) if a.len() == b.len() => {
            for (key, value) in a {
                close(value, b.get(key).ok_or("missing audit key")?, tolerance)?;
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (x, y) in a.iter().zip(b) {
                close(x, y, tolerance)?;
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (
                a.as_f64().ok_or("invalid audit number")?,
                b.as_f64().ok_or("invalid reported number")?,
            );
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > tolerance {
                return Err(format!("audit number mismatch: {a} != {b}").into());
            }
        }
        _ if a == b => {}
        _ => return Err("audit structure mismatch".into()),
    }
    Ok(())
}

pub fn validate(model: &Model, task: &data::Task, report: &Value) -> Result<Value> {
    let (side, orbits, vocab) = (model.shape.side, model.shape.orbits, model.shape.vocab);
    let (cells, q) = (side * side, 2 * orbits - 1);
    let class = &model.decoder.orbit_class;
    if orbits < 2
        || [side, q, vocab].iter().any(|x| !(2..=193).contains(x))
        || vocab != task.vocab
        || class.len() != orbits
        || class.iter().any(|&k| k >= vocab)
        || model.latent_half.len() != cells * orbits
        || model.output.len() != cells * vocab
        || model.latent_half.iter().any(|x| !x.is_finite())
        || model.output.iter().any(|x| !x.is_finite())
    {
        return Err("invalid raw paired tables or decoder".into());
    }
    let budgets = match report["case"]["budget"].as_str() {
        Some("rank_one") => (side as f64, side as f64),
        _ => return Err("invalid raw budget".into()),
    };
    if (model.radius, model.nuclear_budget) != budgets {
        return Err("raw budget mismatch".into());
    }
    // Reflect every row to all q latent states; check it is a probability vector
    // and that summing it over the decoder fibers gives the saved output row.
    let state = |cell: usize, z: usize| model.latent_half[cell * orbits + z.min(q - z)];
    let mut decoding_defect = 0_f64;
    for cell in 0..cells {
        let mut decoded = vec![0.; vocab];
        let mut mass = 0.;
        for z in 0..q {
            let p = state(cell, z);
            if !(-1e-10..=1. + 1e-10).contains(&p) {
                return Err("raw latent entry is not a probability".into());
            }
            mass += p;
            decoded[class[z.min(q - z)]] += p;
        }
        if (mass - 1.).abs() > 1e-10 {
            return Err("raw latent row does not sum to one".into());
        }
        for (a, b) in decoded.iter().zip(&model.output[cell * vocab..]) {
            decoding_defect = decoding_defect.max((a - b).abs());
        }
    }
    if decoding_defect > 1e-12 {
        return Err("saved output table is not the decoded latent table".into());
    }
    let mut diagnostics = serde_json::Map::new();
    for (split, examples) in &task.splits {
        let (mut ce, mut correct, mut transitions, mut transition_correct, mut fit) =
            (0., 0, 0, 0, 0_f64);
        let (mut counts, mut hits) = (vec![0; vocab], vec![0; vocab]);
        for e in examples {
            if e.tokens.len() != 2
                || e.targets.len() != 2
                || e.targets[0] != -100
                || e.targets[1] < 0
                || e.changes.len() != 2
                || !e.prompt.is_empty()
                || e.field.is_some()
                || e.tokens.iter().any(|t| *t < 0 || *t as usize >= side)
                || e.targets[1] as usize >= vocab
            {
                return Err("unsupported raw audit example".into());
            }
            let y = e.targets[1] as usize;
            let start = (e.tokens[1] as usize + side * e.tokens[0] as usize) * vocab;
            let logits = &model.output[start..start + vocab];
            let mut best = 0;
            for c in 1..vocab {
                if logits[c].total_cmp(&logits[best]).is_gt() {
                    best = c;
                }
            }
            let hit = usize::from(best == y);
            correct += hit;
            counts[y] += 1;
            hits[y] += hit;
            if e.changes[1] {
                transitions += 1;
                transition_correct += hit;
            }
            // Every original class enters the softmax, with or without a latent state.
            ce += logits.iter().map(|x| x.exp()).sum::<f64>().ln() - logits[y];
            for (c, &p) in logits.iter().enumerate() {
                fit = fit.max((p - f64::from(c == y)).abs());
            }
        }
        let count = examples.len();
        if count == 0 {
            return Err("empty raw audit split".into());
        }
        let accuracy = correct as f64 / count as f64;
        let classes: serde_json::Map<String, Value> = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(c, &count)| {
                (
                    c.to_string(),
                    json!({"name":c.to_string(),"count":count,"correct":hits[c],
                    "accuracy":hits[c] as f64/count as f64}),
                )
            })
            .collect();
        let balanced = counts
            .iter()
            .zip(&hits)
            .filter(|(count, _)| **count > 0)
            .map(|(&count, &hit)| hit as f64 / count as f64)
            .sum::<f64>()
            / classes.len() as f64;
        let transition = if transitions > 0 {
            Some(transition_correct as f64 / transitions as f64)
        } else {
            None
        };
        let metrics = json!({"loss":ce/count as f64,"example_loss":ce/count as f64,
            "sequence_accuracy":accuracy,"token_accuracy":accuracy,"balanced_accuracy":balanced,
            "transition_accuracy":transition,"correct":correct,"exact":correct,"targets":count,
            "target_count":count,"examples":count,"example_count":count,
            "transition_count":transitions,"classes":classes});
        close(&metrics, &report["metrics"][split], 1e-10)?;
        if split == "train" {
            close(&json!(fit), &report["fit_defect"], 1e-10)?;
            let ideal = (std::f64::consts::E + vocab as f64 - 1.).ln() - 1.;
            close(
                &json!(ce / count as f64 - ideal),
                &report["ideal_ce_excess"],
                1e-10,
            )?;
        }
        diagnostics.insert(
            split.clone(),
            json!({"examples":count,"correct":correct,"ce":ce/count as f64,"one_hot_defect":fit}),
        );
    }
    // All q - 1 nonconstant frequencies, one complex matrix at a time, by direct
    // sums over all q latent states of the reflected table.
    let mut norms = vec![];
    let mut imaginary = 0_f64;
    for f in 1..q {
        let phases: Vec<_> = (0..q)
            .map(|z| C::from_polar(1., std::f64::consts::TAU * ((f * z) % q) as f64 / q as f64))
            .collect();
        let mut matrix = vec![C::default(); cells];
        for (cell, entry) in matrix.iter_mut().enumerate() {
            for (z, phase) in phases.iter().enumerate() {
                *entry += state(cell, z) * phase;
            }
            imaginary = imaginary.max(entry.im.abs());
        }
        let values = linalg::svd(&matrix, side)?.values;
        let (operator, nuclear) = (values[0], values.iter().sum::<f64>());
        if operator > model.radius * (1. + 1e-9) || nuclear > model.nuclear_budget * (1. + 1e-9) {
            return Err("raw spectral bound violation".into());
        }
        norms.push(json!({"operator":operator,"nuclear":nuclear}));
    }
    if imaginary > 1e-9 {
        return Err("raw moments of the reflected table are not real".into());
    }
    let reported = report["moment_norms"]
        .as_array()
        .filter(|x| x.len() == orbits - 1)
        .ok_or("missing reported moment norms")?;
    for f in 1..q {
        close(&norms[f - 1], &reported[f.min(q - f) - 1], 1e-7)?;
    }
    Ok(
        json!({"raw_latent_coordinate_audit":true,"frequencies":q-1,"shared_lapack_svd":true,
        "interval_certificate":false,"largest_decoding_defect":decoding_defect,
        "largest_imaginary_moment_part":imaginary,"diagnostics":diagnostics}),
    )
}
