//! Training-only candidate elimination. Every candidate and conflict is saved;
//! raw categorical and affine-coordinate compatibility are recorded separately.
//! A deterministic first-compatible selector is not an unseen-target theorem.
use crate::{
    coordinates::{Fit, Record},
    program::{Atom, Program},
    *,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decoder {
    pub roles: [Option<usize>; 2],
    pub quantities: BTreeMap<usize, usize>,
    pub quantity_fit: Option<Fit>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Learned {
    pub program: Program,
    pub decoder: Decoder,
    pub geometry_error: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conflict {
    pub first_record: usize,
    pub second_record: usize,
    pub expected: Option<usize>,
    pub observed: usize,
    pub atom: Atom,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub program: Program,
    pub observations: usize,
    pub scalar_work: u64,
    pub max_buffer_symbols: usize,
    pub conflict: Option<Conflict>,
    pub geometry_error: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub input: Input,
    pub coordinates: Vec<f64>,
    pub roles: program::Roles,
    pub attempts: Vec<Attempt>,
    pub retained: Vec<Learned>,
    pub categorical_selection: Option<usize>,
    pub geometry_selection: Option<usize>,
    pub scalar_work: u64,
    pub search_forecast: u64,
    pub buffer_forecast: u64,
}

impl Decoder {
    /// Copy completion is an explicit identity prior in both arms. Only the
    /// geometry arm extrapolates an observed affine quantity decoder.
    pub fn center(&self, coordinates: &[f64], atom: Atom, geometry: bool) -> (f64, bool) {
        let decoded = match atom {
            Atom::Role(r) => self.roles.get(r).copied().flatten().map(|c| coordinates[c]),
            Atom::Copy(c) => coordinates.get(c).copied(),
            Atom::Quantity(q) => {
                if geometry {
                    self.quantity_fit
                        .as_ref()
                        .map(|fit| fit.slope * (q as f64 / 128.) + fit.intercept)
                } else {
                    self.quantities.get(&q).map(|&c| coordinates[c])
                }
            }
        };
        match decoded.filter(|x| x.is_finite() && x.abs() <= 4096.) {
            Some(x) => (x, true),
            None => (coordinates[0], false),
        }
    }
}

fn prepared(
    input: &Input,
    coordinates: &[f64],
) -> Result<(program::Roles, Vec<Program>, u64, u64)> {
    coordinates::validate_coordinates(coordinates)?;
    if coordinates.len() != input.vocabulary
        || input.context > 4096
        || input
            .observations
            .iter()
            .any(|r| r.label >= input.vocabulary || r.tokens.len() > input.context)
    {
        return Err("training labels, context or supplied geometry mismatch".into());
    }
    let roles = program::infer(input)?;
    let bank = program::catalogue(&roles)?;
    let n = input.observations.len() as u128;
    let v = input.vocabulary as u128;
    // Complete candidate execution, a separate recount/render interpreter,
    // categorical/coordinate fit and all-class residual checks. Every rejected
    // candidate is reserved as though it reads every original record.
    let row_work: u128 = input
        .observations
        .iter()
        .map(|r| {
            let l = r.tokens.len() as u128;
            256 * (l * l + v + l + 1) * (1 + r.tokens.len().ilog2() as u128) + 64 * v
        })
        .sum();
    let work = roles.scalar_work_forecast as u128 + bank.len() as u128 * (row_work + 256 * v * v);
    let max_prefix = input
        .observations
        .iter()
        .map(|r| r.tokens.len())
        .max()
        .unwrap() as u128;
    let bytes = 512 * 1024 * 1024
        + 64 * (input
            .observations
            .iter()
            .map(|r| r.tokens.len() as u128 * 8 + 32)
            .sum::<u128>()
            + bank.len() as u128 * (2048 + 64 * v + 128 * n.min(max_prefix))
            + n * 64);
    if work > 128_000_000_000_000 || bytes > 56 * 1024 * 1024 * 1024 {
        return Err("aggregate search work or memory forecast refused".into());
    }
    Ok((roles, bank, work as u64, bytes as u64))
}

pub fn forecast(input: &Input, coordinates: &[f64]) -> Result<(u64, u64)> {
    let (_, _, work, bytes) = prepared(input, coordinates)?;
    Ok((work, bytes))
}

pub fn run(input: Input, coordinates: Vec<f64>) -> Result<Report> {
    let (roles, bank, work, bytes) = prepared(&input, &coordinates)?;
    let v = input.vocabulary as u128;
    let mut out = Report {
        input,
        coordinates,
        roles,
        attempts: vec![],
        retained: vec![],
        categorical_selection: None,
        geometry_selection: None,
        scalar_work: 0,
        search_forecast: work,
        buffer_forecast: bytes,
    };
    for program in bank {
        let mut decoder = Decoder {
            roles: [None, None],
            quantities: BTreeMap::new(),
            quantity_fit: None,
        };
        let mut role_rows = [0; 2];
        let mut quantity_rows = BTreeMap::new();
        let mut numeric = vec![];
        let mut attempt = Attempt {
            program: program.clone(),
            observations: 0,
            scalar_work: 0,
            max_buffer_symbols: 0,
            conflict: None,
            geometry_error: None,
        };
        for (id, row) in out.input.observations.iter().enumerate() {
            let value = program.run(&row.tokens);
            attempt.observations += 1;
            attempt.scalar_work += value.scalar_work;
            attempt.max_buffer_symbols = attempt.max_buffer_symbols.max(value.buffer_symbols);
            let (expected, first) = match value.atom {
                Atom::Role(r) if r < 2 => {
                    if decoder.roles[r].is_none() {
                        decoder.roles[r] = Some(row.label);
                        role_rows[r] = id;
                    }
                    (decoder.roles[r], role_rows[r])
                }
                Atom::Role(_) => (None, id),
                Atom::Copy(c) => (Some(c), id),
                Atom::Quantity(q) => {
                    numeric.push(Record {
                        id,
                        feature: q as f64 / 128.,
                        label: row.label,
                    });
                    (
                        Some(*decoder.quantities.entry(q).or_insert(row.label)),
                        *quantity_rows.entry(q).or_insert(id),
                    )
                }
            };
            if expected != Some(row.label) {
                attempt.conflict = Some(Conflict {
                    first_record: first,
                    second_record: id,
                    expected,
                    observed: row.label,
                    atom: value.atom,
                });
                break;
            }
        }
        if attempt.conflict.is_none() {
            attempt.scalar_work += (64 * v * numeric.len() as u128 + 256 * v * v) as u64;
            if !numeric.is_empty() {
                match Fit::new(out.coordinates.clone(), &numeric) {
                    Ok(fit) => decoder.quantity_fit = Some(fit),
                    Err(error) => attempt.geometry_error = Some(error.to_string()),
                }
            }
            // Interpret all retained training rows independently, before any
            // candidate is eligible for either arm or subsequent CE fitting.
            for row in &out.input.observations {
                let length = row.tokens.len() as u64;
                attempt.scalar_work += 128
                    * (length * length + length + out.input.vocabulary as u64 + 1)
                    * (1 + length.ilog2() as u64);
                if program.run(&row.tokens).atom != program.interpret(&row.tokens) {
                    return Err("producer aggregate disagrees with independent interpreter".into());
                }
            }
            let index = out.retained.len();
            out.categorical_selection.get_or_insert(index);
            if attempt.geometry_error.is_none() {
                out.geometry_selection.get_or_insert(index);
            }
            out.retained.push(Learned {
                program,
                decoder,
                geometry_error: attempt.geometry_error.clone(),
            });
        }
        out.scalar_work += attempt.scalar_work;
        out.attempts.push(attempt);
    }
    if out.scalar_work > out.search_forecast {
        return Err("aggregate constructor exceeded its complete work reserve".into());
    }
    Ok(out)
}

pub fn replay(saved: &Report) -> Result<()> {
    if *saved != run(saved.input.clone(), saved.coordinates.clone())? {
        return Err("saved aggregate candidate search differs from replay".into());
    }
    Ok(())
}
