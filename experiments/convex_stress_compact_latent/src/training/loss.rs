//! The user's forward-only minibatch priority,2026-11-16. Minibatches retain
//! the original record order and scalar CE additions. Only the final total is
//! divided by the original count; an unequal tail is never equally weighted
//! with a full batch. Forward-only retains its existing zero-update policy.
use super::*;
use std::ops::Range;

pub fn loss(head: &Head, records: &[Record], work: &mut Work) -> Result<f64> {
    loss_source(head, Source::Expanded(records), 0..records.len(), work)
}
fn mean(
    head: &Head,
    records: Source<'_>,
    ranges: impl Iterator<Item = Range<usize>>,
    count: usize,
    work: &mut Work,
) -> Result<f64> {
    if count == 0 {
        return Err("empty ordinary CE batch".into());
    }
    records.check_shape(head.shape())?;
    let mut total = 0.;
    for range in ranges {
        records.visit(range, work, |row, work| {
            Domain::record(head.shape(), row)?;
            total += ce(&head.logits(&row.states, work)?, row.label);
            Ok(())
        })?;
    }
    Ok(total / count as f64)
}
pub fn loss_source(
    head: &Head,
    records: Source<'_>,
    range: Range<usize>,
    work: &mut Work,
) -> Result<f64> {
    let count = range.len();
    mean(head, records, std::iter::once(range), count, work)
}
pub fn loss_minibatches(
    head: &Head,
    records: Source<'_>,
    batch_size: usize,
    work: &mut Work,
) -> Result<f64> {
    if !(1..=65536).contains(&batch_size) {
        return Err("invalid forward minibatch size".into());
    }
    let ranges = (0..records.len())
        .step_by(batch_size)
        .map(|first| first..(first + batch_size).min(records.len()));
    mean(head, records, ranges, records.len(), work)
}
pub(super) fn endpoint(
    head: &Head,
    records: Source<'_>,
    config: &Config,
    work: &mut Work,
) -> Result<f64> {
    if config.mode == Mode::Forward {
        loss_minibatches(head, records, config.batch_size, work)
    } else {
        loss_source(head, records, 0..records.len(), work)
    }
}
