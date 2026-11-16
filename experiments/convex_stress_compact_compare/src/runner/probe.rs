//! Read the complete verified probe without allocating and cloning a generic
//! serde_json::Value tree. All candidate IDs, original row counts, digests and
//! training-only metadata are preserved and checked before model preparation.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Report {
    pub selection: pairs::selection::Selection,
    pub work: joint::Work,
    pub protocol_sha256: String,
    pub expanded_training_records_allocated: usize,
    pub heldout_pools_read: usize,
    pub accuracy_measurements: usize,
    pub training_updates: usize,
}
fn digest_valid(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
impl Report {
    pub(crate) fn validate(&self) -> Result<()> {
        let s = &self.selection;
        if self.expanded_training_records_allocated != 0
            || self.heldout_pools_read != 0
            || self.accuracy_measurements != 0
            || self.training_updates != 0
            || self.protocol_sha256 != pairs::digest(pairs::PROTOCOL.as_bytes())
            || self.work.limit != pairs::MAX_WORK
            || self.work.used > self.work.limit
            || self.work.used != s.operations
            || s.family != pairs::FAMILY
            || s.family_sha256 != family_digest()
            || s.total_candidates != pairs::PROGRAMS
            || s.retained.len().checked_add(s.conflicts.len()) != Some(pairs::PROGRAMS)
            || s.records == 0
            || s.records > 65536
            || !digest_valid(&s.original_ordered_training_sha256)
            || !digest_valid(&s.audit.dictionary_sha256)
            || !digest_valid(&s.audit.program_major_states_sha256)
        {
            return Err("complete typed probe metadata differs from its frozen producer".into());
        }
        pairs::family::Evaluator::new(s.vocabulary)?;
        let states = s.vocabulary.div_ceil(4) * 4;
        if s.audit.known_rows.checked_add(s.audit.unknown_rows) != Some(s.retained.len() * states)
            || s.audit.independently_checked_states != s.retained.len() as u64 * s.records as u64
            || s.audit.independently_checked_conflicts != s.conflicts.len()
            || s.audit.head_columns_bound > s.retained.len() * (states / 4) * s.vocabulary
            || !s.retained.windows(2).all(|x| x[0] < x[1])
            || !s.conflicts.windows(2).all(|x| x[0].program < x[1].program)
        {
            return Err("typed probe audit counts or original candidate order changed".into());
        }
        let mut present = vec![false; pairs::PROGRAMS];
        for &id in &s.retained {
            let seen = present
                .get_mut(id)
                .ok_or("retained candidate out of range")?;
            if *seen {
                return Err("duplicate retained candidate".into());
            }
            *seen = true;
        }
        for conflict in &s.conflicts {
            let seen = present
                .get_mut(conflict.program)
                .ok_or("conflicting candidate out of range")?;
            if *seen
                || conflict.first_record >= conflict.second_record
                || conflict.second_record >= s.records
                || conflict.state >= s.vocabulary
            {
                return Err("invalid or duplicate original conflict witness".into());
            }
            *seen = true;
        }
        if present.iter().any(|x| !x) {
            return Err("probe omits a candidate".into());
        }
        Ok(())
    }
}

pub fn read(path: &Path) -> Result<pairs::selection::Selection> {
    let file = fs::File::open(path)?;
    if file.metadata()?.len() > PROBE_LIMIT {
        return Err("complete probe report exceeds its declared 128 MiB limit".into());
    }
    let report: Report = serde_json::from_reader(std::io::BufReader::new(file))?;
    report.validate()?;
    Ok(report.selection)
}
