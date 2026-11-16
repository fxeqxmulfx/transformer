//! The exact CM65 dictionary and indices from reduced_latent at88fb15dc.
//! Admission counts actual dictionary capacities, independently of the old
//! dense-tensor forecast. Validation checks every canonical index in place.
use crate::{memory, records::Source, *};
pub use legacy::domain::Domain;
use std::mem::size_of;

pub fn allocation_bytes(d: &Domain) -> u128 {
    size_of::<Domain>() as u128
        + d.labels.capacity() as u128 * size_of::<Option<usize>>() as u128
        + d.offsets.capacity() as u128 * size_of::<usize>() as u128
        + d.unknown.capacity() as u128 * size_of::<(usize, usize)>() as u128
        + d.known.capacity() as u128 * size_of::<usize>() as u128
}
pub fn dictionary_bytes(shape: Shape, label_capacity: usize, unknown: usize) -> Result<u128> {
    shape.validate()?;
    if label_capacity < shape.rows() || unknown > shape.rows() {
        return Err("invalid complete dictionary allocation counts".into());
    }
    shape
        .programs
        .checked_add(
            unknown
                .checked_mul(shape.classes)
                .ok_or("logical parameter count overflow")?,
        )
        .ok_or("logical parameter count overflow")?;
    Ok(size_of::<Domain>() as u128
        + label_capacity as u128 * size_of::<Option<usize>>() as u128
        + shape.rows() as u128 * size_of::<usize>() as u128
        + unknown as u128 * size_of::<(usize, usize)>() as u128
        + shape.programs as u128 * size_of::<usize>() as u128)
}
fn header(shape: Shape, labels: &[Option<usize>], minimum: f64, maximum: f64) -> Result<()> {
    shape.validate()?;
    if labels.len() != shape.rows()
        || labels.iter().flatten().any(|&y| y >= shape.classes)
        || !minimum.is_finite()
        || !maximum.is_finite()
        || minimum <= 0.
        || minimum > maximum
        || maximum > 64.
    {
        return Err("invalid reduced domain dictionary or mass bounds".into());
    }
    Ok(())
}
fn columns(shape: Shape, labels: &[Option<usize>]) -> Result<usize> {
    let count: u128 = labels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|block| {
            if block.iter().any(Option::is_none) {
                shape.classes as u128
            } else {
                4.min(shape.classes) as u128
            }
        })
        .sum();
    Ok(usize::try_from(count)?)
}
pub fn validate(d: &Domain, work: &mut Work) -> Result<()> {
    header(d.shape, &d.labels, d.minimum_mass, d.maximum_mass)?;
    dictionary_bytes(d.shape, d.labels.capacity(), d.unknown.len())?;
    memory::remaining(96_000_000_000, allocation_bytes(d))?;
    if d.offsets.len() != d.shape.rows() || d.known.len() != d.shape.programs {
        return Err("reduced domain index lengths differ from dictionary".into());
    }
    work.charge(32 * d.shape.rows() as u128)?;
    let mut unknown = 0;
    for p in 0..d.shape.programs {
        let mut known = 0;
        for s in 0..d.shape.states {
            let at = p * d.shape.states + s;
            if d.labels[at].is_some() {
                known += 1;
                if d.offsets[at] != usize::MAX {
                    return Err("known row has a free coordinate offset".into());
                }
            } else {
                if d.unknown.get(unknown) != Some(&(p, s))
                    || d.offsets[at] != d.shape.programs + unknown * d.shape.classes
                {
                    return Err("reduced domain indices differ from dictionary".into());
                }
                unknown += 1;
            }
        }
        if d.known[p] != known {
            return Err("known row count differs from dictionary".into());
        }
    }
    if unknown != d.unknown.len() || d.head_columns_bound != columns(d.shape, &d.labels)? {
        return Err("unknown count or logical head bound differs from dictionary".into());
    }
    Ok(())
}
pub fn from_labels(
    shape: Shape,
    labels: Vec<Option<usize>>,
    minimum: f64,
    maximum: f64,
    limit: u64,
    work: &mut Work,
) -> Result<Domain> {
    header(shape, &labels, minimum, maximum)?;
    work.charge(32 * shape.rows() as u128)?;
    let count = labels.iter().filter(|x| x.is_none()).count();
    let forecast = dictionary_bytes(shape, labels.capacity(), count)?;
    memory::remaining(limit, forecast)?;
    let head_columns_bound = columns(shape, &labels)?;
    let mut offsets = memory::vector(shape.rows(), limit)?;
    offsets.resize(shape.rows(), usize::MAX);
    let mut unknown = memory::vector(count, limit)?;
    let mut known = memory::vector(shape.programs, limit)?;
    known.resize(shape.programs, 0);
    for (i, label) in labels.iter().enumerate() {
        let (p, s) = (i / shape.states, i % shape.states);
        if label.is_some() {
            known[p] += 1;
        } else {
            offsets[i] = shape.programs + unknown.len() * shape.classes;
            unknown.push((p, s));
        }
    }
    let out = Domain {
        shape,
        labels,
        offsets,
        unknown,
        known,
        minimum_mass: minimum,
        maximum_mass: maximum,
        head_columns_bound,
    };
    if allocation_bytes(&out) != forecast {
        return Err("actual dictionary capacities differ from admission".into());
    }
    Ok(out)
}
pub fn from_source(
    shape: Shape,
    records: Source<'_>,
    minimum: f64,
    maximum: f64,
    limit: u64,
    work: &mut Work,
) -> Result<Domain> {
    records.check_shape(shape)?;
    if records.is_empty() {
        return Err("empty original training records".into());
    }
    let label_room = memory::remaining(limit, records.scratch_bytes() + 4096)?;
    let mut labels = memory::vector(shape.rows(), label_room)?;
    labels.resize(shape.rows(), None);
    work.charge(16 * shape.rows() as u128 + 8 * records.len() as u128 * shape.programs as u128)?;
    records.visit(0..records.len(), work, |row, _| {
        Domain::record(shape, row)?;
        for (p, &s) in row.states.iter().enumerate() {
            let slot = &mut labels[p * shape.states + s];
            if slot.is_some_and(|y| y != row.label) {
                return Err("conflicting candidate requires exact zero-block elimination".into());
            }
            *slot = Some(row.label);
        }
        Ok(())
    })?;
    from_labels(shape, labels, minimum, maximum, limit, work)
}

#[cfg(test)]
mod tests;
