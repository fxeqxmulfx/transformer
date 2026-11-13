//! Independent signed-wide arithmetic on all original equations and labels.
use super::*;
pub fn check(request: &Request, out: &Report) -> Result<()> {
    let (names, problem, bytes) = request.problem()?;
    if out.request != *request
        || out.names != names
        || out.buffer_preflight_bytes != bytes
        || out.field_audit != finite_system::check(&problem, &out.field)?
        || out.work.field != out.field.work
        || out.work.ce_gradients != 0
        || out.raw_label_identification
    {
        return Err("source, field certificate, bounds or scope differs".into());
    }
    let (nr, nc) = (names.rows.len(), names.columns.len());
    let mut work = Work {
        field: out.field.work.clone(),
        ..Default::default()
    };
    let mut expected = None;
    let mut conflict = None;
    let mut anchor = None;
    let mut inverse = None;
    let mut reason = out.field.reason.clone();
    if let Some(space) = &out.field.space {
        if space.particular.iter().any(|&x| x != 0) {
            return Err("homogeneous particular solution changed".into());
        }
        if space.directions.len() != 1 {
            reason = "no_nonconstant_direction".into();
        } else {
            let raw = &space.directions[0];
            anchor = (0..names.classes.len()).find(|&k| raw[nr + nc + k] != 0);
            if let Some(a) = anchor {
                let q = request.prime as i128;
                inverse = (1..request.prime).find(|&b| {
                    work.inverse_tests += 1;
                    (b as i128 * raw[nr + nc + a] as i128).rem_euclid(q) == 1
                });
                let inv = inverse.ok_or("missing independent field inverse")? as i128;
                let point: Vec<_> = raw
                    .iter()
                    .map(|&x| (inv * x as i128).rem_euclid(q) as u16)
                    .collect();
                work.normalization_products = point.len();
                let code = Code {
                    prime: request.prime,
                    names: names.clone(),
                    rows: point[..nr].into(),
                    columns: point[nr..nr + nc].into(),
                    classes: point[nr + nc..].into(),
                };
                for (index, &[cell, label]) in request.input.observations.iter().enumerate() {
                    work.observation_checks += 1;
                    let i = names
                        .rows
                        .iter()
                        .position(|&i| i == cell % request.input.n)
                        .ok_or("original row absent")?;
                    let j = names
                        .columns
                        .iter()
                        .position(|&j| j == cell / request.input.n)
                        .ok_or("original column absent")?;
                    let c = names
                        .classes
                        .iter()
                        .position(|&c| c == label)
                        .ok_or("original raw class absent")?;
                    if (point[i] as i128 + point[nr + j] as i128 - point[nr + nc + c] as i128)
                        .rem_euclid(q)
                        != 0
                    {
                        conflict = Some(index);
                        break;
                    }
                }
                if conflict.is_some() {
                    reason = "original_observation_conflict".into();
                } else {
                    reason = "admitted_nonconstant_code_with_possible_aliases".into();
                    expected = Some(code);
                }
            } else {
                reason = "input_only_direction".into();
            }
        }
    }
    let mut fibers = vec![];
    if let Some(code) = &expected {
        fibers = vec![vec![]; request.prime as usize];
        for (&name, &value) in names.classes.iter().zip(&code.classes) {
            fibers[value as usize].push(name);
        }
    }
    if out.code != expected
        || out.conflict != conflict
        || out.anchor_class != anchor
        || out.inverse != inverse
        || out.reason != reason
        || out.work != work
        || out.code_fibers != fibers
    {
        return Err("actual original-equation code/audit/work mismatch".into());
    }
    Ok(())
}
