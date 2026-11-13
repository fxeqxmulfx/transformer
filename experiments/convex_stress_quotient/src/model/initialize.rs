use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Initialization {
    pub model: Model,
    pub counts: Vec<usize>,
    pub totals: Vec<usize>,
    pub observed_increments: usize,
    pub logarithms: usize,
    pub phase_cells: usize,
    pub ce_gradients: usize,
}
pub fn initialize(report: &Report) -> Result<Initialization> {
    crate::check(&report.request, report)?;
    let code = report
        .code
        .as_ref()
        .ok_or("no admitted code to materialize")?;
    let input = &report.request.input;
    let (n, q, v) = (input.n, code.prime as usize, input.vocabulary);
    memory_bound(n, q, v)?;
    let (mut rows, mut columns) = (vec![0; n], vec![0; n]);
    let (mut known_rows, mut known_columns) = (vec![false; n], vec![false; n]);
    for (&name, &z) in code.names.rows.iter().zip(&code.rows) {
        rows[name] = z;
        known_rows[name] = true;
    }
    for (&name, &z) in code.names.columns.iter().zip(&code.columns) {
        columns[name] = z;
        known_columns[name] = true;
    }
    // Missing axes get a total model value, never an identification claim.
    let latent: Vec<_> = (0..n * n)
        .map(|cell| ((rows[cell % n] as usize + columns[cell / n] as usize) % q) as u16)
        .collect();
    let (mut counts, mut totals) = (vec![0usize; q * v], vec![0usize; q]);
    for &[cell, label] in &input.observations {
        let z = latent[cell] as usize;
        counts[z * v + label] += 1;
        totals[z] += 1;
    }
    let mut logarithms = 0;
    let coefficients = counts
        .iter()
        .enumerate()
        .map(|(k, &count)| {
            if count == 0 {
                return 0.;
            }
            logarithms += 2;
            24. + ((count as f64).ln() - (totals[k / v] as f64).ln()).max(-24.)
        })
        .collect();
    let out = Initialization {
        model: Model {
            n,
            q,
            vocabulary: v,
            rows,
            columns,
            known_rows,
            known_columns,
            latent,
            coefficients,
            radius: 32. * v as f64,
        },
        counts,
        totals,
        observed_increments: input.observations.len(),
        logarithms,
        phase_cells: n * n,
        ce_gradients: 0,
    };
    out.model.observations(&input.observations)?;
    Ok(out)
}

/// Rebuild coordinates with signed-wide arithmetic and counts from original
/// labels. The log-frequency initializer is a proposal, not a CE optimum proof.
pub fn check_initialization(report: &Report, out: &Initialization) -> Result<()> {
    crate::check(&report.request, report)?;
    let code = report
        .code
        .as_ref()
        .ok_or("initializer attached to refused code")?;
    let input = &report.request.input;
    let model = &out.model;
    model.observations(&input.observations)?;
    if (model.n, model.q, model.vocabulary) != (input.n, code.prime as usize, input.vocabulary) {
        return Err("initializer dimensions differ from original input".into());
    }
    for i in 0..input.n {
        let r = code.names.rows.iter().position(|&x| x == i);
        let c = code.names.columns.iter().position(|&x| x == i);
        if model.known_rows[i] != r.is_some()
            || model.known_columns[i] != c.is_some()
            || model.rows[i] != r.map_or(0, |k| code.rows[k])
            || model.columns[i] != c.map_or(0, |k| code.columns[k])
        {
            return Err("known-axis mask or phase extension differs".into());
        }
    }
    let (mut counts, mut totals) = (vec![0; model.q * model.vocabulary], vec![0; model.q]);
    for &[cell, label] in &input.observations {
        let z = (model.rows[cell % input.n] as i128 + model.columns[cell / input.n] as i128)
            .rem_euclid(code.prime as i128) as usize;
        counts[z * model.vocabulary + label] += 1;
        totals[z] += 1;
    }
    if counts != out.counts
        || totals != out.totals
        || out.ce_gradients != 0
        || out.observed_increments != input.observations.len()
        || out.phase_cells != input.n * input.n
        || out.logarithms != 2 * counts.iter().filter(|&&x| x != 0).count()
    {
        return Err("forward counts or construction work differs".into());
    }
    for (k, &count) in counts.iter().enumerate() {
        let expected = if count == 0 {
            0.
        } else {
            24. + (count as f64 / totals[k / model.vocabulary] as f64)
                .ln()
                .max(-24.)
        };
        if (model.coefficients[k] - expected).abs() > 1e-12 {
            return Err(
                "saved initializer is not its original raw-label frequency proposal".into(),
            );
        }
    }
    Ok(())
}
