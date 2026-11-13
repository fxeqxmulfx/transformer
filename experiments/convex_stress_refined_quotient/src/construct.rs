//! Observed-only CM31 refinement. Most frequent raw class per latent code;
//! ties use its earliest categorical cell, so class renaming preserves ties.
use crate::{
    model::{Block, phase},
    *,
};
use std::collections::BTreeMap;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Initialization {
    pub model: Model,
    pub bounds: limits::Bounds,
    pub observed_states: usize,
    pub default_states: usize,
    pub unsupported_defaults: usize,
    pub complex_products: usize,
    pub ce_gradients: usize,
}
pub fn initialize(discovery: &quotient::Report) -> Result<Initialization> {
    let input = &discovery.request.input;
    limits::dimensions(input.n, discovery.request.prime as usize, input.vocabulary)?;
    if input.observations.is_empty() || input.observations.len() > input.n * input.n {
        return Err("refinement requires bounded nonempty observations".into());
    }
    quotient::check(&discovery.request, discovery)?;
    let code = discovery
        .code
        .as_ref()
        .ok_or("refused code cannot be refined")?;
    let (n, q, v) = (input.n, code.prime as usize, input.vocabulary);
    let (mut rows, mut columns) = (vec![None; n], vec![None; n]);
    for (&name, &z) in code.names.rows.iter().zip(&code.rows) {
        rows[name] = Some(z);
    }
    for (&name, &z) in code.names.columns.iter().zip(&code.columns) {
        columns[name] = Some(z);
    }
    let mut groups = vec![vec![]; q];
    for &[cell, label] in &input.observations {
        let z = (rows[cell % n].ok_or("observed row missing code")? as usize
            + columns[cell / n].ok_or("observed column missing code")? as usize)
            % q;
        groups[z].push((cell, label));
    }
    for g in &mut groups {
        g.sort_unstable();
    }
    let bounds = limits::shape(v, &groups.iter().map(|g| g.len() + 1).collect::<Vec<_>>())?;
    let mut model = Model {
        n,
        vocabulary: v,
        rows,
        columns,
        blocks: vec![],
        table: vec![0.; bounds.states * v],
        coefficients: vec![0.; bounds.real_coefficients],
        radius: 64. * bounds.states as f64,
    };
    let (mut offset, mut unsupported) = (0, 0);
    for g in groups {
        let mut counts = BTreeMap::<usize, (usize, usize)>::new();
        for &(cell, label) in &g {
            let entry = counts.entry(label).or_insert((0, cell));
            entry.0 += 1;
        }
        let default_label = counts
            .iter()
            .max_by(|a, b| a.1.0.cmp(&b.1.0).then_with(|| b.1.1.cmp(&a.1.1)))
            .map(|(&label, _)| label)
            .unwrap_or(0);
        unsupported += usize::from(g.is_empty());
        let states = g.len() + 1;
        for (s, &(_, label)) in g.iter().enumerate() {
            model.table[(offset + s) * v + label] = 24.;
        }
        model.table[(offset + g.len()) * v + default_label] = 24.;
        model.blocks.push(Block {
            observed: g.into_iter().map(|s| s.0).collect(),
            default_label,
            offset,
            code: (0..states)
                .flat_map(|s| (0..states).map(move |f| phase(states, s, f)))
                .collect(),
        });
        offset += states;
    }
    model.materialize();
    model.validate()?;
    Ok(Initialization {
        observed_states: input.observations.len(),
        default_states: q,
        unsupported_defaults: unsupported,
        complex_products: bounds.phase_entries * v,
        ce_gradients: 0,
        model,
        bounds,
    })
}
pub fn check(discovery: &quotient::Report, out: &Initialization) -> Result<()> {
    if initialize(discovery)? != *out {
        return Err("refined initialization does not replay".into());
    }
    let samples = &discovery.request.input.observations;
    let actual = out.model.audit(samples)?;
    if actual.train_errors != 0 {
        return Err("actual refined initializer misclassifies an observed label".into());
    }
    for &[cell, label] in samples {
        let logits = audit::complex_logits(&out.model, out.model.location(cell));
        if logits
            .iter()
            .enumerate()
            .any(|(c, &z)| (z - if c == label { 24. } else { 0. }).abs() > 1e-7)
        {
            return Err("actual refined initializer fails recorded raw margins".into());
        }
    }
    Ok(())
}
