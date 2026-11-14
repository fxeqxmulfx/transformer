//! Import every original CM34 offer, preserving the reviewed selector and
//! explicit refusals. Selected histories receive fresh numerical replay.
use super::*;
use crate::selection as comparison;
pub(crate) mod imported;
pub(crate) mod outcomes;
pub(crate) struct Bundle {
    pub record: Value,
    pub offers: Vec<selection::Offer>,
}
pub(crate) fn candidates(
    pin: &Reference,
    stage: &Value,
    task: &str,
    x: &inputs::Inputs,
) -> Result<Bundle> {
    let record = imported::case(pin, stage, task, x)?;
    let mut offers = vec![];
    if record["status"] == "complete" {
        for saved in record["result"]["stages"]
            .as_array()
            .ok_or("complete raw stages omitted")?
        {
            let s = imported::load(pin, task, &record, saved, x, false)?;
            if let Some(head) = &s.attempt.head {
                offers.extend(outcomes::offers(task, s.prime, &s.sha, head, s.eligible)?);
            }
        }
    }
    Ok(Bundle { record, offers })
}
