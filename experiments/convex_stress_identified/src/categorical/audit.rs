use super::*;

fn refused(out: &Report, reason: &str) -> Result<()> {
    if out.certified || out.kernel.is_some() || out.reason != reason {
        return Err("categorical refusal supplied a kernel or completion certificate".into());
    }
    Ok(())
}
/// Rebuild the exact observation matrix and shifts with signed i128 arithmetic;
/// the independent ring checker then proves its actual zero-kernel condition.
pub fn check(input: &Request, out: &Report) -> Result<()> {
    let (_, bound) = input.validate()?;
    if &out.input != input || out.buffer_bound_bytes != bound || out.work.ce_gradients != 0 {
        return Err(
            "categorical certificate input, forecast or gradient accounting differs".into(),
        );
    }
    let n = input.data.n;
    let mut classes = vec![];
    let mut indices = vec![];
    let mut seen_r = vec![0; n];
    let mut seen_c = vec![0; n];
    for &[cell, label] in &input.data.observations {
        let c = match classes.iter().position(|&v| v == label) {
            Some(c) => c,
            None => {
                classes.push(label);
                classes.len() - 1
            }
        };
        indices.push(c);
        seen_r[cell % n] += 1;
        seen_c[cell / n] += 1;
    }
    if classes != out.classes
        || out.anchor_row != input.data.observations[0][0] % n
        || out.missing_rows != (0..n).filter(|&i| seen_r[i] == 0).collect::<Vec<_>>()
        || out.missing_columns != (0..n).filter(|&i| seen_c[i] == 0).collect::<Vec<_>>()
    {
        return Err("categorical observed partition or actual missing axes differ".into());
    }
    let p = &input.proposal;
    let collision =
        (0..classes.len()).find_map(|i| (0..i).find(|&j| p.codes[i] == p.codes[j]).map(|j| [j, i]));
    if out.collision != collision {
        return Err("categorical code collision witness differs".into());
    }
    if collision.is_some() {
        if out.conflict.is_some()
            || out.normalized.is_some()
            || out.anchor_class.is_some()
            || out.work != Work::default()
        {
            return Err("nonbijective code contains further work".into());
        }
        return refused(out, "nonbijective_code");
    }
    let q = classes.len() as i128;
    let conflict = input
        .data
        .observations
        .iter()
        .zip(&indices)
        .position(|(&[cell, _], &c)| {
            (p.rows[cell % n] as i128 + p.columns[cell / n] as i128 - p.codes[c] as i128)
                .rem_euclid(q)
                != 0
        });
    if out.conflict != conflict
        || out.work.observation_checks != conflict.map_or(input.data.observations.len(), |i| i + 1)
    {
        return Err("original categorical equation/conflict/work differs".into());
    }
    if conflict.is_some() {
        if out.normalized.is_some()
            || out.anchor_class.is_some()
            || out.work.normalized_coordinates != 0
        {
            return Err("conflicting observations supplied a normalized endpoint".into());
        }
        return refused(out, "observed_equation_conflict");
    }
    let row0 = p.rows[out.anchor_row] as i128;
    let code0 = p.codes[0] as i128;
    let mut point: Vec<_> = p
        .rows
        .iter()
        .map(|&v| (v as i128 - row0).rem_euclid(q) as u16)
        .collect();
    point.extend(
        p.columns
            .iter()
            .map(|&v| (v as i128 + row0 - code0).rem_euclid(q) as u16),
    );
    point.extend(
        p.codes
            .iter()
            .map(|&v| (v as i128 - code0).rem_euclid(q) as u16),
    );
    let anchor = (0..classes.len())
        .find(|&i| point[2 * n + i] == 1)
        .ok_or("missing actual class anchor")?;
    if out.normalized.as_ref() != Some(&point)
        || out.anchor_class != Some(anchor)
        || out.work.normalized_coordinates != point.len()
    {
        return Err("actual shifted joint vector or unit anchor differs".into());
    }
    if !out.missing_rows.is_empty() || !out.missing_columns.is_empty() {
        return refused(out, "missing_axis_refusal");
    }
    let mut matrix = vec![];
    for (k, &[cell, _]) in input
        .data
        .observations
        .iter()
        .take(input.retained_equations)
        .enumerate()
    {
        let mut row = Vec::with_capacity(point.len());
        for v in 0..point.len() {
            let a = i128::from(v == cell % n) + i128::from(v == n + cell / n)
                - i128::from(v == 2 * n + indices[k]);
            row.push(a.rem_euclid(q) as u16);
        }
        matrix.push(row);
    }
    for v in [out.anchor_row, 2 * n, 2 * n + anchor] {
        matrix.push((0..point.len()).map(|j| u16::from(v == j)).collect());
    }
    let problem = ring::Problem {
        modulus: q as u16,
        variables: point.len(),
        rows: matrix,
        policy: input.policy.clone(),
    };
    let kernel = out
        .kernel
        .as_ref()
        .ok_or("missing attempted categorical kernel certificate")?;
    ring::check(&problem, kernel)?;
    let reason = if kernel.complete {
        "unique_within_bijective_cyclic_model".into()
    } else {
        format!("kernel_{}", kernel.reason)
    };
    if out.certified != kernel.complete || out.reason != reason {
        return Err("categorical certificate status differs from actual row replay".into());
    }
    Ok(())
}
