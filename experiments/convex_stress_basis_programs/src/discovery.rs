//! Training-only finite discovery. Original multiplicities determine empirical
//! coordinate moments and ordinary complete-head CE. These estimates initialize
//! joint parameters; they are not additional identifying anchors (CM60/CM61).
use crate::{input::Input, program::Program, *};
use basis::data::Example;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frequency {
    pub atom: Atom,
    pub label: usize,
    pub count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub program: Program,
    pub frequencies: Vec<Frequency>,
    pub unit: Parameters,
    pub budget: f64,
    pub exact: bool,
    pub maximum_anchor_residual: f64,
    pub training_ce: f64,
    pub training_correct: u64,
    pub records: u64,
}
impl Candidate {
    pub fn categorical(&self, atom: Atom, coordinates: &[f64]) -> Option<f64> {
        if let Atom::Copy(c) = atom {
            return coordinates.get(c).copied();
        }
        let (mut n, mut total) = (0u64, 0.);
        for f in self.frequencies.iter().filter(|f| f.atom == atom) {
            n += f.count;
            total += f.count as f64 * coordinates[f.label];
        }
        (n > 0).then(|| total / n as f64)
    }
    pub fn structural_anchor(&self, atom: Atom) -> bool {
        matches!(atom, Atom::Role(_))
            && self.frequencies.iter().filter(|f| f.atom == atom).count() == 1
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attempt {
    pub program: Program,
    pub visited: usize,
    pub error: Option<String>,
    pub exact: Option<bool>,
    pub training_ce: Option<f64>,
    pub maximum_anchor_residual: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Discovery {
    pub input: Input,
    pub coordinates: Vec<f64>,
    pub attempts: Vec<Attempt>,
    pub selected: Option<Candidate>,
    pub scalar_forecast: u64,
    pub buffer_forecast: u64,
    pub retained_original_records: usize,
}

fn estimate(
    program: Program,
    frequencies: Vec<Frequency>,
    coordinates: &[f64],
) -> Result<Candidate> {
    let mut unit = [0., 0., 0., 0., 1., 1.];
    let quantity: Vec<_> = frequencies
        .iter()
        .filter_map(|f| {
            if let Atom::Quantity(q) = f.atom {
                Some((q as f64 / 128., coordinates[f.label], f.count as f64))
            } else {
                None
            }
        })
        .collect();
    let n: f64 = quantity.iter().map(|r| r.2).sum();
    if n > 0. {
        let x = quantity.iter().map(|r| r.0 * r.2).sum::<f64>() / n;
        let y = quantity.iter().map(|r| r.1 * r.2).sum::<f64>() / n;
        let variance: f64 = quantity.iter().map(|r| r.2 * (r.0 - x).powi(2)).sum();
        if variance > 1e-18 {
            unit[0] = quantity
                .iter()
                .map(|r| r.2 * (r.0 - x) * (r.1 - y))
                .sum::<f64>()
                / variance;
        }
        unit[1] = y - unit[0] * x;
    }
    for role in 0..2 {
        let rows: Vec<_> = frequencies
            .iter()
            .filter(|f| f.atom == Atom::Role(role))
            .collect();
        let n: u64 = rows.iter().map(|f| f.count).sum();
        if n > 0 {
            unit[role + 2] = rows
                .iter()
                .map(|f| f.count as f64 * coordinates[f.label])
                .sum::<f64>()
                / n as f64;
        }
    }
    let mut residual: f64 = 0.;
    for f in &frequencies {
        let h = f.atom.features(coordinates)?;
        residual = residual.max(
            (h.iter().zip(unit).map(|(x, b)| x * b).sum::<f64>() - coordinates[f.label]).abs(),
        );
    }
    let exact = residual <= 1e-10;
    let p: serde_json::Value = serde_json::from_str(PROTOCOL)?;
    let mass = p["fit"][if exact { "exact_mass" } else { "relaxed_mass" }]
        .as_f64()
        .ok_or("missing initializer mass")?;
    let budget = (joint::charge(&[unit]) + 8.).max(16.);
    if !budget.is_finite()
        || budget > 1e6
        || joint::charge(&[unit]) * mass > p["fit"]["radius"].as_f64().unwrap()
    {
        return Err("observed-moment initializer exceeds declared complete budgets".into());
    }
    let head = joint::head::Head::new(coordinates.to_vec(), vec![unit.map(|x| x * mass)])?;
    let (mut loss, mut correct, mut records) = (0., 0u64, 0u64);
    for f in &frequencies {
        let z = head.logits(&[f.atom.features(coordinates)?])?;
        loss += f.count as f64 * joint::ce(&z, f.label);
        correct += f.count * u64::from(joint::argmax(&z) == f.label);
        records += f.count;
    }
    Ok(Candidate {
        program,
        frequencies,
        unit,
        budget,
        exact,
        maximum_anchor_residual: residual,
        training_ce: loss / records as f64,
        training_correct: correct,
        records,
    })
}

pub fn run(vocabulary: usize, rows: &[Example]) -> Result<Discovery> {
    let input = input::inspect(vocabulary, rows)?;
    let bank = program::catalogue(&input)?;
    let scalar = bank.len() as u128
        * (128 * (input.prefix_positions as u128 + input.records as u128)
            + 4096 * (256 * vocabulary as u128 + 8192));
    let buffer = 256 * 1024u128.pow(2) + 64 * (4096 * 512 + bank.len() as u128 * 8192);
    if scalar > 600_000_000_000_000 || buffer > joint::MAX_BYTES {
        return Err("complete discovery forecast refused".into());
    }
    let coordinates = previous::observations::coordinates(vocabulary)?;
    let mut out = Discovery {
        input,
        coordinates,
        attempts: vec![],
        selected: None,
        scalar_forecast: scalar as u64,
        buffer_forecast: buffer as u64,
        retained_original_records: 0,
    };
    for program in bank {
        let mut frequencies = BTreeMap::<(Atom, usize), u64>::new();
        let (mut visited, mut error) = (0, None);
        for row in input::observations(rows) {
            visited += 1;
            let Some(atom) = program.run(row.tokens) else {
                error = Some("undefined on an original training prefix".into());
                break;
            };
            if let Atom::Copy(c) = atom
                && c != row.label
            {
                error = Some("observed label contradicts supplied identity-copy prior".into());
                break;
            }
            *frequencies.entry((atom, row.label)).or_default() += 1;
            if frequencies.len() > 4096 {
                error = Some("original atom/label histogram cap exceeded".into());
                break;
            }
        }
        let mut attempt = Attempt {
            program: program.clone(),
            visited,
            error,
            exact: None,
            training_ce: None,
            maximum_anchor_residual: None,
        };
        if attempt.error.is_none() {
            let f = frequencies
                .into_iter()
                .map(|((atom, label), count)| Frequency { atom, label, count })
                .collect();
            match estimate(program, f, &out.coordinates) {
                Ok(candidate) => {
                    if candidate.records != out.input.records as u64 {
                        return Err("candidate omitted original observations".into());
                    }
                    attempt.exact = Some(candidate.exact);
                    attempt.training_ce = Some(candidate.training_ce);
                    attempt.maximum_anchor_residual = Some(candidate.maximum_anchor_residual);
                    let better = out.selected.as_ref().is_none_or(|old| {
                        (candidate.exact && !old.exact)
                            || (candidate.exact == old.exact
                                && candidate.training_ce < old.training_ce - 1e-12)
                    });
                    if better {
                        out.selected = Some(candidate);
                    }
                }
                Err(e) => attempt.error = Some(e.to_string()),
            }
        }
        out.attempts.push(attempt);
    }
    if let Some(chosen) = &out.selected {
        let mut independent = BTreeMap::<(Atom, usize), u64>::new();
        for row in input::observations(rows) {
            let atom = chosen
                .program
                .interpret(row.tokens)
                .ok_or("independent retained program undefined")?;
            if chosen.program.run(row.tokens) != Some(atom) {
                return Err("independent original-prefix interpreter differs".into());
            }
            *independent.entry((atom, row.label)).or_default() += 1;
            out.retained_original_records += 1;
        }
        let actual: Vec<_> = independent
            .into_iter()
            .map(|((atom, label), count)| Frequency { atom, label, count })
            .collect();
        if actual != chosen.frequencies || out.retained_original_records != out.input.records {
            return Err("independent complete retained histogram differs".into());
        }
    }
    Ok(out)
}
