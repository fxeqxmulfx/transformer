use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Initialization {
    pub model: Model,
    pub assigned_states: usize,
    pub complex_code_entries: usize,
    pub complex_head_additions: usize,
    pub ce_gradients: usize,
    pub buffer_preflight_bytes: u64,
}
pub fn initialize(report: &discovery::Report) -> Result<Initialization> {
    discovery::check(&report.request, report)?;
    if report.reason.is_some() {
        return Err("refused discovery cannot initialize a complete model".into());
    }
    let states: usize = report
        .groups
        .iter()
        .map(|g| g.decoder.as_ref().map_or(0, |d| d.values.len() + 1))
        .sum();
    let entries: usize = report
        .groups
        .iter()
        .map(|g| {
            g.decoder
                .as_ref()
                .map_or(0, |d| (d.values.len() + 1).pow(2))
        })
        .sum();
    let x = &report.request.input;
    let bytes = memory_bound(x.vocabulary, states, entries)?;
    let mut model = Model {
        vocabulary: x.vocabulary,
        context: x.context,
        blocks: vec![],
        coefficients: vec![0.; 2 * states * x.vocabulary],
        radius: 64. * states as f64,
    };
    let mut rows = 0;
    for g in &report.groups {
        let decoder = g
            .decoder
            .clone()
            .ok_or("selected query lacks its actual decoder")?;
        let n = decoder.values.len() + 1;
        let code: Vec<_> = (0..n)
            .flat_map(|s| (0..n).map(move |f| phase(n, s, f)))
            .collect();
        for state in 0..n {
            let label = decoder.labels.get(state).copied().unwrap_or(0);
            for f in 0..n {
                let h = code[state * n + f].conj() * 24.;
                let row = (rows + 2 * f) * x.vocabulary;
                model.coefficients[row + label] += h.re;
                model.coefficients[row + x.vocabulary + label] += h.im;
            }
        }
        model.blocks.push(Block {
            query: g.query,
            decoder,
            first_row: rows,
            code,
        });
        rows += 2 * n;
    }
    model.validate()?;
    Ok(Initialization {
        model,
        assigned_states: states,
        complex_code_entries: entries,
        complex_head_additions: entries,
        ce_gradients: 0,
        buffer_preflight_bytes: bytes,
    })
}
pub fn check(report: &discovery::Report, initial: &Initialization) -> Result<()> {
    if initialize(report)? != *initial {
        return Err("saved causal Fourier initialization differs".into());
    }
    initial.model.code_audit_forecast()?;
    // Independent inversion check on every actual categorical state, including
    // the explicit unknown-value default. No target answer function is used.
    for (i, b) in initial.model.blocks.iter().enumerate() {
        for state in 0..b.states() {
            let expected = b.decoder.labels.get(state).copied().unwrap_or(0);
            let logits = initial.model.logits_state(i, state);
            if logits
                .iter()
                .enumerate()
                .any(|(c, &z)| (z - if c == expected { 24. } else { 0. }).abs() > 1e-7)
            {
                return Err(
                    "actual complex forward pass does not reproduce observed-state initialization"
                        .into(),
                );
            }
        }
    }
    Ok(())
}
