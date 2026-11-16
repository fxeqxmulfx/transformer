//! Prospective work policy for the complete CM70 comparison. This explicit
//! limit changes admission only, never the original charged operations,
//! candidate family, numerical policy or a frozen producer's limits.
use crate::*;
use joint::{
    Shape,
    training::{Config, Mode},
};

/// Separate from each 40-trillion preparation pass. The largest original
/// minibatch endpoint needs over 54 trillion mandatory update charges alone.
/// Initial/final audits, predictions and checkpointing retain their charges.
pub const ENDPOINT_LIMIT: u64 = 256_000_000_000_000;

/// Reserve checkpoint work before training; release it without resetting the
/// used counter. Work::new belongs to the older 40-trillion policy, so this
/// prospective constructor deliberately installs the documented new ceiling.
pub fn endpoint(reserve: u64) -> Result<Work> {
    let limit = ENDPOINT_LIMIT
        .checked_sub(reserve)
        .filter(|&n| n > 0)
        .ok_or("endpoint reserve exhausts work limit")?;
    Ok(Work { used: 0, limit })
}
pub fn release_reserve(work: &mut Work) -> Result<()> {
    if work.limit == 0 || work.limit > ENDPOINT_LIMIT || work.used > work.limit {
        return Err("invalid prospective endpoint work counter".into());
    }
    work.limit = ENDPOINT_LIMIT;
    Ok(())
}
/// A lower bound, not a running-time forecast or a promise that an endpoint
/// fits. Includes the mandatory charges in gradient, Target::step, project,
/// Plan::prepare, feasibility/KKT, Head::build and parameter distance. Excludes
/// scalar solver iterations, record contractions, initialization and saving.
pub fn update_lower_bound(
    shape: Shape,
    unknown: usize,
    head_columns: usize,
    records: usize,
    config: &Config,
) -> Result<u64> {
    shape.validate()?;
    config.validate()?;
    if unknown > shape.rows()
        || head_columns > shape.rows() / 4 * shape.classes
        || !(1..=65536).contains(&records)
    {
        return Err("invalid complete update work dimensions".into());
    }
    let batch = if config.mode == Mode::Mini {
        config.batch_size
    } else {
        records
    };
    let steps = config.epochs as u128 * records.div_ceil(batch) as u128;
    let parameters = shape.programs as u128 + unknown as u128 * shape.classes as u128;
    let mandatory = (100 + 8 * shape.classes.ilog2() as u128) * parameters
        + 112 * shape.rows() as u128
        + 160 * head_columns as u128;
    Ok(u64::try_from(steps * mandatory)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_largest_minibatch_cannot_fit_old_work_limit_without_cutting_updates() {
        let shape = Shape {
            programs: 962806,
            states: 196,
            classes: 193,
        };
        let lower =
            update_lower_bound(shape, 7324366, 840996226, 18528, &pairs::policy(Mode::Mini))
                .unwrap();
        assert_eq!(lower, 54_949_599_919_456);
        assert!(lower > pairs::MAX_WORK && lower < ENDPOINT_LIMIT);
        assert_eq!(
            update_lower_bound(
                shape,
                7324366,
                840996226,
                18528,
                &pairs::policy(Mode::Forward)
            )
            .unwrap(),
            0
        );
        let mut work = endpoint(100).unwrap();
        work.charge(lower as u128).unwrap();
        release_reserve(&mut work).unwrap();
        assert_eq!(work.used, lower);
        work.charge((ENDPOINT_LIMIT - lower) as u128).unwrap();
        assert!(work.charge(1).is_err());
        assert_eq!(work.used, ENDPOINT_LIMIT);
        assert!(endpoint(ENDPOINT_LIMIT).is_err());
        work.limit += 1;
        assert!(release_reserve(&mut work).is_err());
    }
    #[test]
    fn bound_is_below_measured_training_and_preserves_real_tail_updates() {
        use joint::{Record, domain, records::Packed, training};
        let shape = Shape {
            programs: 2,
            states: 4,
            classes: 3,
        };
        let rows: Vec<_> = (0..130)
            .map(|i| Record {
                states: vec![i % 2, i % 2],
                label: i % 2,
            })
            .collect();
        let mut work = endpoint(0).unwrap();
        let packed = Packed::from_records(shape, &rows, 1_000_000, &mut work).unwrap();
        let domain = domain::from_source(
            shape,
            joint::records::Source::Packed(&packed),
            1.,
            32.,
            1_000_000,
            &mut work,
        )
        .unwrap();
        for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
            let config = pairs::policy(mode);
            let bound = update_lower_bound(
                shape,
                domain.unknown.len(),
                domain.head_columns_bound,
                rows.len(),
                &config,
            )
            .unwrap();
            let trained = training::train(&domain, &rows, config, 1_000_000, &mut work).unwrap();
            assert!(bound < trained.operations);
            if mode == Mode::Mini {
                assert_eq!(trained.steps.len(), 2);
                assert_eq!(trained.steps[1].first_record, 127);
                assert_eq!(trained.steps[1].end_record_exclusive, 130);
            }
        }
    }
}
