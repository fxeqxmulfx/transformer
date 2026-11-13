//! Independently check the complete grouped basis before binding its vectors
//! to original raw labels. No fabricated raw-label completeness certificate.
use super::*;
pub fn check(request: &Request, out: &Report) -> Result<()> {
    let names = request.names()?;
    if out.request != *request
        || out.raw_label_identification
        || out.unknown_target_compatibility_proved
    {
        return Err("readout request or conditional identification scope changed".into());
    }
    let mut work = Work {
        partition_entries: request.readout.len(),
        ..Work::default()
    };
    if request.answers != usize::from(request.prime) {
        if out.grouped.is_some()
            || out.raw_code.is_some()
            || out.work != work
            || out.reason != "answer_code_cardinality_mismatch"
        {
            return Err("cardinality refusal was altered".into());
        }
        return Ok(());
    }
    work.grouped_observations = request.input.observations.len();
    let grouped = out
        .grouped
        .as_ref()
        .ok_or("missing grouped discovery outcome")?;
    kernel::discovery::check(&request.grouped_request(), grouped)?;
    let mut reason = grouped.reason.as_str();
    let mut expected = None;
    if let Some(code) = &grouped.code {
        let (comparisons, collision) = collisions(code);
        work.group_pair_comparisons = comparisons;
        if collision {
            reason = "incompatible_observed_group_codes";
        } else {
            let raw = expand(request, names, code)?;
            work.expanded_coordinates = raw.dimension * raw.classes.len();
            for &[cell, label] in &request.input.observations {
                work.observed_vector_checks += 1;
                let query = raw.query(cell % request.input.n, cell / request.input.n);
                if query.as_deref() != raw.label(label)
                    || raw.label(label) != code.label(request.readout[label])
                {
                    return Err("original raw observation has an invalid grouped binding".into());
                }
            }
            expected = Some(raw);
            reason = "declared_readout_complete_kernel";
        }
    }
    if out.reason != reason || out.raw_code != expected || out.work != work {
        return Err("expanded raw vectors, refusals or construction costs changed".into());
    }
    Ok(())
}
