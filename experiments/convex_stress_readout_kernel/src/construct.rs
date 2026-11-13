//! CM35 ports kernel/construct.rs at166bebc3, changing only the discovery
//! binding to checked expanded raw codes. Fourier phases, raw margin24, original
//! observed states, bounds and transforms are unchanged. No grouped-label CE.
use crate::*;
use kernel::{model::Block, states};
fn phase(n: usize, state: usize, frequency: usize) -> C {
    C::from_polar(
        1. / (n as f64).sqrt(),
        std::f64::consts::TAU * ((state * frequency) % n) as f64 / n as f64,
    )
}
pub fn initialize(discovery: &discovery::Report) -> Result<Initialization> {
    discovery::check(&discovery.request, discovery)?;
    let input = &discovery.request.input;
    let code = discovery
        .raw_code
        .as_ref()
        .ok_or("refused kernel cannot initialize a complex head")?;
    let encoding = states::Encoding::new(input.n, input.vocabulary, code)?;
    let mut groups = vec![vec![]; encoding.keys.len()];
    for &[cell, label] in &input.observations {
        let z = encoding.cell_blocks[cell]
            .ok_or("original observation lacks an active vector state")?;
        groups[z].push((cell, label));
    }
    for group in &mut groups {
        group.sort_unstable();
    }
    let bounds = limits::shape(
        input.vocabulary,
        &groups.iter().map(|g| g.len() + 1).collect::<Vec<_>>(),
    )?;
    let mut model = Model {
        n: input.n,
        vocabulary: input.vocabulary,
        encoding,
        blocks: vec![],
        table: vec![0.; bounds.states * input.vocabulary],
        coefficients: vec![0.; bounds.real_coefficients],
        radius: 64. * bounds.states as f64,
    };
    let mut offset = 0;
    for (z, group) in groups.into_iter().enumerate() {
        let default_label = model.encoding.default_label(z);
        let states = group.len() + 1;
        for (s, &(_, label)) in group.iter().enumerate() {
            model.table[(offset + s) * input.vocabulary + label] = 24.;
        }
        model.table[(offset + group.len()) * input.vocabulary + default_label] = 24.;
        model.blocks.push(Block {
            observed: group.iter().map(|&(cell, _)| cell).collect(),
            observed_labels: group.iter().map(|&(_, label)| label).collect(),
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
        default_states: model.encoding.keys.len(),
        unmatched_states: model
            .encoding
            .classes
            .iter()
            .filter(|c| c.is_empty())
            .count(),
        unknown_axis_cells: model.encoding.unknown_axis_cells,
        complex_products: bounds.phase_entries * input.vocabulary,
        vector_additions: model.encoding.vector_additions,
        ce_gradients: 0,
        model,
        bounds,
    })
}
pub fn check(discovery: &discovery::Report, out: &Initialization) -> Result<()> {
    if initialize(discovery)? != *out {
        return Err("kernel head initialization does not replay".into());
    }
    let samples = &discovery.request.input.observations;
    let actual = out.model.audit(samples)?;
    if actual.train_errors != 0 {
        return Err("actual kernel initializer misclassifies an observed label".into());
    }
    for &[cell, label] in samples {
        let logits = audit::complex_logits(&out.model, out.model.location(cell));
        if logits
            .iter()
            .enumerate()
            .any(|(c, &z)| (z - if c == label { 24. } else { 0. }).abs() > 1e-7)
        {
            return Err("actual kernel initializer fails recorded raw margins".into());
        }
    }
    Ok(())
}
