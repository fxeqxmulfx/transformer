use super::*;

pub fn run(input: Request) -> Result<Report> {
    let (classes, bound) = input.validate()?;
    let n = input.data.n;
    let q = classes.len() as u16;
    let mut rows = vec![false; n];
    let mut columns = vec![false; n];
    for &[cell, _] in &input.data.observations {
        rows[cell % n] = true;
        columns[cell / n] = true;
    }
    let mut out = Report {
        anchor_row: input.data.observations[0][0] % n,
        input,
        classes,
        buffer_bound_bytes: bound,
        anchor_class: None,
        missing_rows: (0..n).filter(|&i| !rows[i]).collect(),
        missing_columns: (0..n).filter(|&j| !columns[j]).collect(),
        collision: None,
        conflict: None,
        normalized: None,
        kernel: None,
        certified: false,
        reason: "started".into(),
        work: Work::default(),
    };
    let p = &out.input.proposal;
    for i in 0..p.codes.len() {
        for j in 0..i {
            if p.codes[i] == p.codes[j] {
                out.collision = Some([j, i]);
                out.reason = "nonbijective_code".into();
                return Ok(out);
            }
        }
    }
    let indices: Vec<_> = out
        .input
        .data
        .observations
        .iter()
        .map(|s| out.classes.iter().position(|c| *c == s[1]).unwrap())
        .collect();
    for (k, (&[cell, _], &c)) in out.input.data.observations.iter().zip(&indices).enumerate() {
        out.work.observation_checks += 1;
        if (p.rows[cell % n] as u32 + p.columns[cell / n] as u32) % q as u32 != p.codes[c] as u32 {
            out.conflict = Some(k);
            out.reason = "observed_equation_conflict".into();
            return Ok(out);
        }
    }
    let a = p.rows[out.anchor_row] as u32;
    let b = p.codes[0] as u32;
    let modulus = q as u32;
    let mut point: Vec<_> = p
        .rows
        .iter()
        .map(|&x| ((x as u32 + modulus - a) % modulus) as u16)
        .collect();
    point.extend(
        p.columns
            .iter()
            .map(|&x| ((x as u32 + a + modulus - b) % modulus) as u16),
    );
    point.extend(
        p.codes
            .iter()
            .map(|&x| ((x as u32 + modulus - b) % modulus) as u16),
    );
    let anchor = (0..q as usize)
        .find(|&c| point[2 * n + c] == 1)
        .ok_or("bijective code lacks unit anchor")?;
    out.work.normalized_coordinates = point.len();
    out.normalized = Some(point);
    out.anchor_class = Some(anchor);
    if !out.missing_rows.is_empty() || !out.missing_columns.is_empty() {
        out.reason = "missing_axis_refusal".into();
        return Ok(out);
    }
    let variables = 2 * n + q as usize;
    let mut matrix = vec![];
    for (&[cell, _], &c) in out
        .input
        .data
        .observations
        .iter()
        .zip(&indices)
        .take(out.input.retained_equations)
    {
        let mut row = vec![0; variables];
        row[cell % n] = 1;
        row[n + cell / n] = 1;
        row[2 * n + c] = q - 1;
        matrix.push(row);
    }
    // Exactly the three rows in Lean anchoredMatrix. Never pin missing axes.
    for v in [out.anchor_row, 2 * n, 2 * n + anchor] {
        let mut row = vec![0; variables];
        row[v] = 1;
        matrix.push(row);
    }
    let kernel = ring::run(ring::Problem {
        modulus: q,
        variables,
        rows: matrix,
        policy: out.input.policy.clone(),
    })?;
    ring::check(&kernel.input, &kernel)?;
    out.certified = kernel.complete;
    out.reason = if out.certified {
        "unique_within_bijective_cyclic_model".into()
    } else {
        format!("kernel_{}", kernel.reason)
    };
    out.kernel = Some(kernel);
    Ok(out)
}
