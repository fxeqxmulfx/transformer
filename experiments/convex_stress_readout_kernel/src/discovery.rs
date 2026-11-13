//! CM35 Factorization/Identification: equal code/answer cardinality and a
//! declared surjective readout. Observed vector separation is checked; it does
//! not establish an unknown target's global additive code or compatibility.
use crate::*;
use kernel::discovery::{Code, Input, Names};
use std::collections::BTreeSet;
mod check;
pub use check::check;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub input: Input,
    /// Fixed metadata for every raw vocabulary label, including unobserved ones.
    pub readout: Vec<usize>,
    pub answers: usize,
    pub prime: u16,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub partition_entries: usize,
    pub grouped_observations: usize,
    pub group_pair_comparisons: usize,
    pub expanded_coordinates: usize,
    pub observed_vector_checks: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub request: Request,
    /// Genuine complete-kernel certificate for the GROUPED observations only.
    pub grouped: Option<kernel::discovery::Report>,
    pub raw_code: Option<Code>,
    pub reason: String,
    pub work: Work,
    pub raw_label_identification: bool,
    pub unknown_target_compatibility_proved: bool,
}
impl Request {
    fn names(&self) -> Result<Vec<usize>> {
        let i = &self.input;
        limits::dimensions(i.n, i.vocabulary)?;
        if !kernel::finite_system::prime(self.prime)
            || !(2..=1024).contains(&self.answers)
            || self.readout.len() != i.vocabulary
            || self.readout.iter().any(|&d| d >= self.answers)
            || self.readout.iter().copied().collect::<BTreeSet<_>>().len() != self.answers
            || i.observations.is_empty()
            || i.observations.len() > i.n * i.n
            || !(1..=8190).contains(&i.policy.equations)
            || i.policy.elimination.nullity > 8
            || i.policy.elimination.scalar_updates > 1_000_000_000
            || i.policy.elimination.row_operations > 2_000_000
        {
            return Err("invalid bounded request or declared surjective readout".into());
        }
        let mut cells = BTreeSet::new();
        let mut names = vec![];
        for &[cell, label] in &i.observations {
            if cell >= i.n * i.n || label >= i.vocabulary || !cells.insert(cell) {
                return Err("invalid or repeated original cell/label".into());
            }
            if !names.contains(&label) {
                names.push(label);
            }
        }
        Ok(names)
    }
    fn grouped_request(&self) -> kernel::discovery::Request {
        kernel::discovery::Request {
            prime: self.prime,
            input: Input {
                n: self.input.n,
                vocabulary: self.answers,
                observations: self
                    .input
                    .observations
                    .iter()
                    .map(|&[cell, label]| [cell, self.readout[label]])
                    .collect(),
                policy: self.input.policy.clone(),
            },
        }
    }
}
fn expand(request: &Request, names: Vec<usize>, code: &Code) -> Result<Code> {
    let raw = Code {
        prime: code.prime,
        dimension: code.dimension,
        names: Names {
            rows: code.names.rows.clone(),
            columns: code.names.columns.clone(),
            classes: names.clone(),
        },
        rows: code.rows.clone(),
        columns: code.columns.clone(),
        classes: names
            .iter()
            .map(|&c| {
                code.label(request.readout[c])
                    .map(<[u16]>::to_vec)
                    .ok_or("observed raw class has no grouped vector")
            })
            .collect::<std::result::Result<_, _>>()?,
    };
    raw.validate(request.input.n, request.input.vocabulary)?;
    Ok(raw)
}
fn collisions(code: &Code) -> (usize, bool) {
    let mut comparisons = 0;
    let mut collision = false;
    for (i, a) in code.classes.iter().enumerate() {
        for b in &code.classes[i + 1..] {
            comparisons += 1;
            collision |= a == b;
        }
    }
    (comparisons, collision)
}
pub fn run(request: Request) -> Result<Report> {
    let names = request.names()?;
    let mut out = Report {
        work: Work {
            partition_entries: request.readout.len(),
            ..Work::default()
        },
        request,
        grouped: None,
        raw_code: None,
        reason: "answer_code_cardinality_mismatch".into(),
        raw_label_identification: false,
        unknown_target_compatibility_proved: false,
    };
    if out.request.answers != usize::from(out.request.prime) {
        return Ok(out);
    }
    out.work.grouped_observations = out.request.input.observations.len();
    let grouped = kernel::discovery::run(out.request.grouped_request())?;
    out.reason = grouped.reason.clone();
    if let Some(code) = &grouped.code {
        let (comparisons, collision) = collisions(code);
        out.work.group_pair_comparisons = comparisons;
        if collision {
            out.reason = "incompatible_observed_group_codes".into();
        } else {
            let raw = expand(&out.request, names, code)?;
            out.work.expanded_coordinates = raw.dimension * raw.classes.len();
            for &[cell, label] in &out.request.input.observations {
                out.work.observed_vector_checks += 1;
                if raw
                    .query(cell % out.request.input.n, cell / out.request.input.n)
                    .as_deref()
                    != raw.label(label)
                {
                    return Err("expanded code fails an original raw observation".into());
                }
            }
            out.raw_code = Some(raw);
            out.reason = "declared_readout_complete_kernel".into();
        }
    }
    out.grouped = Some(grouped);
    Ok(out)
}
pub fn replay(request: &Request, report: &Report) -> Result<()> {
    check(request, report)?;
    if run(request.clone())? != *report {
        return Err("grouped discovery or charged work does not replay".into());
    }
    Ok(())
}
