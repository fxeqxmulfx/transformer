//! Check grouped costs directly from observations. This deliberately does not
//! call the producer's cost-table or minimization routines. Each observation
//! is visited q times, including categories whose chosen phases coincide.
use super::*;

pub(super) fn axis(
    samples: &[Sample],
    old: &State,
    step: &Step,
    n: usize,
    q: usize,
) -> Result<usize> {
    let row = step.kind == Kind::Row;
    let s = &step.state;
    if s.decoder != old.decoder || (row && s.columns != old.columns) || (!row && s.rows != old.rows)
    {
        return Err("free-axis certificate changed fixed coordinates".into());
    }
    let minima = step
        .minimum_costs
        .as_ref()
        .ok_or("missing free-axis minima")?;
    if minima.len() != n {
        return Err("free-axis certificate length mismatch".into());
    }
    let mut groups = vec![vec![]; n];
    for o in samples {
        let (r, c) = (o.cell % n, o.cell / n);
        groups[if row { r } else { c }]
            .push((if row { old.columns[c] } else { old.rows[r] }, o.label));
    }
    let (chosen, previous) = if row {
        (&s.rows, &old.rows)
    } else {
        (&s.columns, &old.columns)
    };
    let mut checks = 0;
    for a in 0..n {
        let mut costs = Vec::with_capacity(q);
        for z in 0..q {
            let mut cost = 0;
            for &(other, label) in &groups[a] {
                checks += 1;
                cost += usize::from(old.decoder[(z + other) % q] != label);
            }
            costs.push(cost);
        }
        if costs[chosen[a]] != minima[a] || costs.iter().any(|&c| c < minima[a]) {
            return Err("a saved phase fails the exhaustive integer minimum certificate".into());
        }
        let expected = if costs[previous[a]] == minima[a] {
            previous[a]
        } else {
            costs.iter().position(|&c| c == minima[a]).unwrap()
        };
        if chosen[a] != expected {
            return Err("free-axis earlier-phase tie policy changed".into());
        }
    }
    if minima.iter().sum::<usize>() != s.errors {
        return Err("grouped minimal costs differ from actual table errors".into());
    }
    Ok(checks)
}

pub(super) fn decoder(
    old: &State,
    step: &Step,
    counts: &[usize],
    classes: &[usize],
    v: usize,
) -> Result<usize> {
    let s = &step.state;
    if s.rows != old.rows || s.columns != old.columns || step.minimum_costs.is_some() {
        return Err("free decoder step changed phase coordinates or certificate".into());
    }
    let mut checks = 0;
    for (z, row) in counts.chunks(v).enumerate() {
        let maximum = *row.iter().max().unwrap();
        checks += v;
        let expected = if row[old.decoder[z]] == maximum {
            old.decoder[z]
        } else {
            *classes
                .iter()
                .find(|&&c| row[c] == maximum)
                .ok_or("unobserved maximum label")?
        };
        if s.decoder[z] != expected {
            return Err("decoder count maximum or earlier-label tie failed".into());
        }
    }
    Ok(checks)
}
