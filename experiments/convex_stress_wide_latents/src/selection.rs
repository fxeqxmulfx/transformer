//! Exact candidate elimination and streamed independent state/dictionary audit.
//! Every original record is visited, with its original ID and multiplicity.
//! No expanded record or model tensor is allocated to compute its forecast.
use crate::{
    family::{Evaluator, programs},
    *,
};
pub use pairs::selection::Conflict;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Audit {
    pub known_rows: usize,
    pub unknown_rows: usize,
    pub head_columns_bound: usize,
    pub dictionary_sha256: String,
    pub program_major_states_sha256: String,
    pub independently_checked_states: u64,
    pub independently_checked_conflicts: usize,
    pub complete_buffer_forecast: Option<u64>,
    pub admission_refusal: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub family: String,
    pub family_sha256: String,
    pub vocabulary: usize,
    pub records: usize,
    pub original_ordered_training_sha256: String,
    pub total_candidates: usize,
    pub retained: Vec<usize>,
    pub conflicts: Vec<Conflict>,
    pub audit: Audit,
    pub operations: u64,
}
impl Selection {
    pub fn shape(&self) -> joint::Shape {
        joint::Shape {
            programs: self.retained.len(),
            states: self.vocabulary.div_ceil(4) * 4,
            classes: self.vocabulary,
        }
    }
}
pub(crate) fn hash_header(hash: &mut Sha256, program: usize, size: usize) {
    hash.update((program as u64).to_le_bytes());
    hash.update((size as u64).to_le_bytes());
}
pub fn select(training: &Training, work: &mut Work) -> Result<Selection> {
    let start = work.used;
    work.charge(16384 * PROGRAMS as u128 + 4096 * training.rows.len() as u128 + 67_108_864)?;
    // Generate the entire bank without inspecting any training label.
    let family = programs();
    let family_sha256 = digest(&serde_json::to_vec(&family)?);
    training.validate()?;
    let evaluator = Evaluator::new(training.vocabulary)?;
    let states = training.vocabulary.div_ceil(4) * 4;
    let (mut retained, mut conflicts) = (Vec::new(), Vec::new());
    let (mut known, mut unknown, mut columns) = (0, 0, 0);
    let (mut dictionary, mut encoded) = (Sha256::new(), Sha256::new());
    for (program, &tree) in family.iter().enumerate() {
        work.charge(1024 * states as u128)?;
        let mut seen: Vec<Option<(usize, u32)>> = vec![None; states];
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
            work.charge(2 * evaluator.reference_cost(tree))?;
            let [a, b, y] = training.rows[witness.first_record];
            let [c, d, z] = training.rows[witness.second_record];
            if y == z
                || evaluator.reference(tree, a as usize, b as usize) != witness.state
                || evaluator.reference(tree, c as usize, d as usize) != witness.state
            {
                return Err("independent wide-family collision witness failed".into());
            }
            conflicts.push(witness);
        } else {
            hash_header(&mut encoded, program, training.rows.len());
            for &[a, b, y] in &training.rows {
                work.charge(512 + evaluator.reference_cost(tree))?;
                let state = evaluator.evaluate(tree, a as usize, b as usize);
                if evaluator.reference(tree, a as usize, b as usize) != state
                    || seen[state].map(|x| x.1) != Some(y)
                {
                    return Err("independent wide-family observed state differs".into());
                }
                encoded.update((state as u32).to_le_bytes());
            }
            hash_header(&mut dictionary, program, states);
            for label in &seen {
                dictionary.update(label.map_or(u32::MAX, |x| x.1).to_le_bytes());
            }
            let observed = seen.iter().filter(|x| x.is_some()).count();
            known += observed;
            unknown += states - observed;
            for block in seen.as_chunks::<4>().0 {
                columns += if block.iter().any(Option::is_none) {
                    training.vocabulary
                } else {
                    4.min(training.vocabulary)
                };
            }
            retained.push(program);
        }
    }
    let shape = joint::Shape {
        programs: retained.len(),
        states,
        classes: training.vocabulary,
    };
    let forecast = if retained.is_empty() {
        Err("empty compatible support in the complete fixed family".into())
    } else {
        shape.admit(training.rows.len(), unknown, columns)
    };
    let (complete_buffer_forecast, admission_refusal) = match forecast {
        Ok(bytes) => (Some(bytes), None),
        Err(e) => (None, Some(e.to_string())),
    };
    let audit = Audit {
        known_rows: known,
        unknown_rows: unknown,
        head_columns_bound: columns,
        dictionary_sha256: hex(&dictionary.finalize()),
        program_major_states_sha256: hex(&encoded.finalize()),
        independently_checked_states: retained.len() as u64 * training.rows.len() as u64,
        independently_checked_conflicts: conflicts.len(),
        complete_buffer_forecast,
        admission_refusal,
    };
    Ok(Selection {
        family: FAMILY.into(),
        family_sha256,
        vocabulary: training.vocabulary,
        records: training.rows.len(),
        original_ordered_training_sha256: digest(&serde_json::to_vec(training)?),
        total_candidates: family.len(),
        retained,
        conflicts,
        audit,
        operations: work.used - start,
    })
}
