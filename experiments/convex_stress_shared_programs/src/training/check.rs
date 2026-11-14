use super::*;

/// Separate bisection projection used only by the checker.
fn bisection(values: &[f64]) -> Vec<f64> {
    let mut low = values.iter().copied().fold(f64::INFINITY, f64::min) - 1.;
    let mut high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for _ in 0..180 {
        let middle = (low + high) / 2.;
        if values.iter().map(|x| (x - middle).max(0.)).sum::<f64>() > 1. {
            low = middle;
        } else {
            high = middle;
        }
    }
    values
        .iter()
        .map(|x| (x - (low + high) / 2.).max(0.))
        .collect()
}

pub fn check(initial: &Model, input: &Input, out: &Outcome) -> Result<()> {
    if run(initial, input, out.policy.clone())? != *out {
        return Err("complete CE trajectory/work replay differs".into());
    }
    let samples = initial.encode(input)?;
    let table = initial.head.independent_table();
    let mut weights = initial.weights.clone();
    for step in &out.steps {
        let mut g = vec![0.; weights.len()];
        let mut loss = 0.;
        for sample in &samples[step.start..step.start + step.count] {
            let z: Vec<f64> = (0..initial.head.vocabulary)
                .map(|c| {
                    weights
                        .iter()
                        .zip(&sample.states)
                        .map(|(&w, &s)| w * table[s][c])
                        .sum()
                })
                .collect();
            loss += ce(&z, sample.label);
            let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mass = z.iter().map(|x| (x - top).exp()).sum::<f64>();
            for (j, &state) in sample.states.iter().enumerate() {
                g[j] += (z
                    .iter()
                    .enumerate()
                    .map(|(c, &v)| (v - top).exp() / mass * table[state][c])
                    .sum::<f64>()
                    - table[state][sample.label])
                    / step.count as f64;
            }
        }
        if (loss / step.count as f64 - step.batch_ce).abs() > 1e-9
            || (g.iter().map(|x| x * x).sum::<f64>().sqrt() - step.gradient_l2).abs() > 1e-8
        {
            return Err("independent ordinary CE gradient/loss differs".into());
        }
        let proposed: Vec<_> = weights
            .iter()
            .zip(g)
            .map(|(&w, d)| w - out.policy.learning_rate * d)
            .collect();
        let projected = bisection(&proposed);
        if projected
            .iter()
            .zip(&step.weights)
            .any(|(a, b)| (a - b).abs() > 1e-8)
        {
            return Err("independent bisection mixture update differs".into());
        }
        weights = step.weights.clone();
    }
    if weights != out.model.weights
        || out.model.head != initial.head
        || out.model.programs != initial.programs
    {
        return Err("fixed complete head/programs or final mixture changed".into());
    }
    Ok(())
}
