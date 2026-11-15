//! Training-only homogeneous compatibility. A positive-mass nullspace witness
//! checks each retained input program; its scaled decoder is not fixed during
//! subsequent CM57 CE. Numerical checks do not establish unknown realizability.
use crate::{
    observations::Observation,
    program::{Atom, Program},
    *,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Compatible {
    pub program: Program,
    pub unit_parameters: Parameters,
    pub budget: f64,
    pub rank: usize,
    pub maximum_anchor_residual: f64,
    pub categorical: Vec<(Atom, usize)>,
    pub categorical_conflicts: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub program: Program,
    pub training_records_available: usize,
    pub error: Option<String>,
    pub rank: Option<usize>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Discovery {
    pub attempts: Vec<Attempt>,
    pub compatible: Vec<Compatible>,
    pub coordinates: Vec<f64>,
    pub scalar_forecast: u64,
    pub buffer_forecast: u64,
}

fn compatible(program: &Program, rows: &[Observation], coordinates: &[f64]) -> Result<Compatible> {
    let mut anchors = vec![[0., 0., 0., 0., 1., -1.]];
    let mut categorical = BTreeMap::new();
    let mut categorical_conflicts = 0;
    for r in rows {
        let atom = program
            .run(&r.tokens)
            .ok_or("program undefined on a training prefix")?;
        let h = atom.features(coordinates)?;
        anchors.push([h[0], h[1], h[2], h[3], h[4], -coordinates[r.label]]);
        if let Some(old) = categorical.get(&atom) {
            if *old != r.label {
                categorical_conflicts += 1;
            }
        } else {
            categorical.insert(atom, r.label);
        }
    }
    let basis = joint::projection::basis(&anchors)?;
    let mut unit = [0., 0., 0., 0., 0., 1.];
    joint::projection::linear(&mut unit, &basis);
    if unit[5] <= 1e-12 {
        return Err("observations force zero program mass".into());
    }
    let mass = unit[5];
    for x in &mut unit {
        *x /= mass;
    }
    let residual = anchors
        .iter()
        .map(|a| {
            joint::projection::dot(a, &unit).abs() / a.iter().map(|x| x.abs()).sum::<f64>().max(1.)
        })
        .fold(0., f64::max);
    if residual > 1e-10 {
        return Err("positive-mass anchor witness failed original-row check".into());
    }
    let budget = (joint::charge(&[unit]) + 8.).max(16.);
    if budget > 1e6 || unit.iter().any(|x| !x.is_finite() || x.abs() > 1e9) {
        return Err("positive-mass witness exceeds the declared budget envelope".into());
    }
    Ok(Compatible {
        program: program.clone(),
        unit_parameters: unit,
        budget,
        rank: basis.len(),
        maximum_anchor_residual: residual,
        categorical: categorical.into_iter().collect(),
        categorical_conflicts,
    })
}

pub fn run(rows: &[Observation], coordinates: Vec<f64>, bank: Vec<Program>) -> Result<Discovery> {
    crate::observations::validate(rows, coordinates.len())?;
    joint::data::coordinates(&coordinates)?;
    if bank.is_empty() || bank.len() > 8192 {
        return Err("fixed candidate bank size refused".into());
    }
    let n = rows.len() as u128;
    let length_work: u128 = rows
        .iter()
        .map(|r| (r.tokens.len() as u128 + 1).pow(2))
        .sum();
    let work = bank.len() as u128 * (length_work * 4096 + n * 4096);
    let bytes = 512 * 1024 * 1024 + 64 * (n * 128 + bank.len() as u128 * (2048 + n * 64));
    if work > 600_000_000_000_000 || bytes > joint::MAX_BYTES {
        return Err("discovery work/storage forecast refused".into());
    }
    let mut out = Discovery {
        attempts: vec![],
        compatible: vec![],
        coordinates,
        scalar_forecast: work as u64,
        buffer_forecast: bytes as u64,
    };
    for program in bank {
        match compatible(&program, rows, &out.coordinates) {
            Ok(value) => {
                out.attempts.push(Attempt {
                    program,
                    training_records_available: rows.len(),
                    rank: Some(value.rank),
                    error: None,
                });
                out.compatible.push(value);
            }
            Err(e) => out.attempts.push(Attempt {
                program,
                training_records_available: rows.len(),
                rank: None,
                error: Some(e.to_string()),
            }),
        }
    }
    if out.compatible.len() > joint::MAX_PROGRAMS {
        return Err("compatible bank exceeds complete-head program cap".into());
    }
    Ok(out)
}
