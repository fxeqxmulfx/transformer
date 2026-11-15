//! Exact CM64 elimination: keep every program without a conflicting observed
//! latent state. Preserve a checked collision witness for every zero block;
//! refuse memory/work overflow instead of taking a fitted-candidate prefix.
use crate::{
    family::{Evaluator, programs},
    *,
};
use joint::{Record, Shape, domain::Domain};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conflict {
    pub program: usize,
    pub first_record: usize,
    pub second_record: usize,
    pub state: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub family: String,
    pub family_sha256: String,
    pub vocabulary: usize,
    pub records: usize,
    pub original_ordered_training_sha256: String,
    pub total_candidates: usize,
    pub retained: Vec<usize>,
    pub conflicts: Vec<Conflict>,
    pub operations: u64,
}
impl Selection {
    pub fn shape(&self) -> Shape {
        Shape {
            programs: self.retained.len(),
            states: self.vocabulary.div_ceil(4) * 4,
            classes: self.vocabulary,
        }
    }
}
pub fn select(training: &Training, work: &mut Work) -> Result<Selection> {
    // The entire family is generated solely from this fixed source before
    // reading a raw training label. No candidate-generation feedback exists.
    let start = work.used;
    work.charge(1024 * PROGRAMS as u128 + 256 * training.rows.len() as u128)?;
    let family = programs();
    let family_sha256 = digest(&serde_json::to_vec(&family)?);
    training.validate()?;
    let evaluator = Evaluator::new(training.vocabulary)?;
    let mut retained = vec![];
    let mut conflicts = vec![];
    for (program, &tree) in family.iter().enumerate() {
        work.charge(4 * training.vocabulary as u128)?;
        let mut seen: Vec<Option<(usize, u32)>> = vec![None; training.vocabulary];
        let mut conflict = None;
        for (index, &[a, b, y]) in training.rows.iter().enumerate() {
            work.charge(256)?;
            let state = evaluator.evaluate(tree, a as usize, b as usize);
            match seen[state] {
                Some((first, label)) if label != y => {
                    conflict = Some(Conflict {
                        program,
                        first_record: first,
                        second_record: index,
                        state,
                    });
                    break;
                }
                None => seen[state] = Some((index, y)),
                _ => {}
            }
        }
        if let Some(witness) = conflict {
            work.charge(32768)?;
            let [a, b, y] = training.rows[witness.first_record];
            let [c, d, z] = training.rows[witness.second_record];
            if y == z
                || evaluator.reference(tree, a as usize, b as usize) != witness.state
                || evaluator.reference(tree, c as usize, d as usize) != witness.state
            {
                return Err("independent zero-block collision witness failed".into());
            }
            conflicts.push(witness);
        } else {
            retained.push(program);
        }
    }
    Ok(Selection {
        family: FAMILY.into(),
        family_sha256,
        vocabulary: training.vocabulary,
        records: training.rows.len(),
        original_ordered_training_sha256: digest(&serde_json::to_vec(training)?),
        total_candidates: family.len(),
        retained,
        conflicts,
        operations: work.used - start,
    })
}

/// Encoding checks every retained program on every original record with the
/// second interpreter. No deduplication or heldout information is involved.
pub fn encode(
    training: &Training,
    selection: &Selection,
    work: &mut Work,
) -> Result<(Domain, Vec<Record>)> {
    training.validate()?;
    let family = programs();
    if selection.family != FAMILY
        || selection.family_sha256 != digest(&serde_json::to_vec(&family)?)
        || selection.vocabulary != training.vocabulary
        || selection.records != training.rows.len()
        || selection.original_ordered_training_sha256 != digest(&serde_json::to_vec(training)?)
    {
        return Err("selected family does not match original training records".into());
    }
    // Recheck completeness, including conflict witnesses, before allocating
    // full joint coefficients. A modified survivor list cannot omit a fit.
    let again = select(training, work)?;
    if again.retained != selection.retained
        || again.conflicts != selection.conflicts
        || selection.total_candidates != family.len()
    {
        return Err("selected family omits or alters an exact candidate block".into());
    }
    let shape = selection.shape();
    shape.admit(training.rows.len())?;
    work.charge(16384 * training.rows.len() as u128 * shape.programs as u128)?;
    let evaluator = Evaluator::new(training.vocabulary)?;
    let mut rows = Vec::with_capacity(training.rows.len());
    for &[a, b, y] in &training.rows {
        let mut states = Vec::with_capacity(shape.programs);
        for &id in &selection.retained {
            let state = evaluator.evaluate(family[id], a as usize, b as usize);
            if evaluator.reference(family[id], a as usize, b as usize) != state {
                return Err("independent retained-state interpretation differs".into());
            }
            states.push(state);
        }
        rows.push(Record {
            states,
            label: y as usize,
        });
    }
    let domain = Domain::new(shape, &rows, 1., 32.)?;
    Ok((domain, rows))
}
