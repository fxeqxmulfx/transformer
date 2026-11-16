use super::*;
use training::Mode;
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
fn fixture() -> (Vec<Record>, Packed, Domain) {
    let rows: Vec<_> = (0..130)
        .map(|i| Record {
            states: vec![i % 4, (i % 4 + 1) % 8],
            label: i % 4,
        })
        .collect();
    let sh = Shape {
        programs: 2,
        states: 8,
        classes: 5,
    };
    let records = Packed::from_records(sh, &rows, 65536, &mut work()).unwrap();
    let domain =
        domain::from_source(sh, Source::Packed(&records), 1., 8., 65536, &mut work()).unwrap();
    (rows, records, domain)
}
fn config(mode: Mode) -> Config {
    Config {
        mode,
        epochs: if mode == Mode::Forward { 0 } else { 1 },
        batch_size: 127,
        learning_rate: 3.,
        initial_mass: 2.,
    }
}
#[test]
fn inputs_external_capacity_runtime_and_process_caps_are_additive() {
    let (_, p, mut d) = fixture();
    d.labels.reserve_exact(200);
    let budget = Budget::new(PROCESS_BYTES, 12345).unwrap();
    assert_eq!(budget.core_bytes(), CORE_BYTES);
    assert_eq!(budget.runtime_bytes(), PROCESS_BYTES - CORE_BYTES);
    assert_eq!(budget.process_bytes(), PROCESS_BYTES);
    assert_eq!(budget.external_bytes(), 12345);
    let owned = domain::allocation_bytes(&d) + p.allocation_bytes();
    assert_eq!(
        budget.model_room(&d, &p).unwrap() as u128,
        CORE_BYTES as u128 - 12345 - owned
    );
    assert!(
        budget
            .after_input_forecast(d.shape, p.len(), d.unknown.len())
            .unwrap()
            > budget.model_room(&d, &p).unwrap()
    );
    assert!(Budget::new(PROCESS_BYTES + 1, 0).is_err());
    assert!(Budget::new(MIN_RUNTIME_BYTES, 0).is_err());
    assert!(Budget::new(PROCESS_BYTES, CORE_BYTES as u128).is_err());
    assert!(budget.remaining(u128::MAX).is_err());
    let small = Budget::new(MIN_RUNTIME_BYTES + 65536, 1024).unwrap();
    assert_eq!(small.runtime_bytes(), MIN_RUNTIME_BYTES);
    assert_eq!(small.remaining(0).unwrap(), 64512);
}
#[test]
fn whole_input_training_matches_the_prior_packed_policies_and_refuses_overcommit() {
    let (rows, records, d) = fixture();
    let old_records = legacy::records::Packed::from_records(d.shape, &rows, &mut work()).unwrap();
    // Charge the test-owned source rows, their spare capacity, and old packed
    // control as external buffers; the production runner uses its own ledger.
    let external = std::mem::size_of_val(&rows) as u128
        + rows.capacity() as u128 * std::mem::size_of::<Record>() as u128
        + rows
            .iter()
            .map(|r| r.states.capacity() as u128 * 8)
            .sum::<u128>()
        + old_records.stored_bytes() as u128
        + std::mem::size_of_val(&old_records) as u128;
    let budget = Budget::new(MIN_RUNTIME_BYTES + 131072, external).unwrap();
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let (mut a, mut b) = (work(), work());
        let new = budget.train(&d, &records, config(mode), &mut b).unwrap();
        let old = legacy::training::train_source(
            &d,
            legacy::records::Source::Packed(&old_records),
            config(mode),
            &mut a,
        )
        .unwrap();
        assert_eq!(a.used, b.used);
        assert_eq!(old.final_ce.to_bits(), new.final_ce.to_bits());
        assert!(
            old.parameters
                .values
                .iter()
                .map(|x| x.to_bits())
                .eq(new.parameters.values().map(|x| x.to_bits()))
        );
        assert_eq!(old.steps.len(), new.steps.len());
    }
    assert!(
        training::train_source(
            &d,
            Source::Packed(&records),
            config(Mode::Mini),
            65536,
            &mut work()
        )
        .is_ok()
    );
    let limited = Budget::new(MIN_RUNTIME_BYTES + 65536, 60000).unwrap();
    assert!(
        limited
            .train(&d, &records, config(Mode::Mini), &mut work())
            .is_err()
    );
}
#[test]
fn pack_and_dictionary_are_bounded_before_model_construction() {
    let sh = Shape {
        programs: 2,
        states: 8,
        classes: 5,
    };
    let labels = [1, 2, 1];
    let budget = Budget::new(MIN_RUNTIME_BYTES + 65536, 1024).unwrap();
    let p = budget
        .pack(sh, &labels, |r, p, _| Ok((r % 2) + p), &mut work())
        .unwrap();
    let d = budget.dictionary(&p, 1., 8., &mut work()).unwrap();
    assert_eq!(d.known, vec![2, 2]);
    let tiny = Budget::new(MIN_RUNTIME_BYTES + 1024, 1000).unwrap();
    let mut calls = 0;
    assert!(
        tiny.pack(
            sh,
            &labels,
            |_, _, _| {
                calls += 1;
                Ok(0)
            },
            &mut work()
        )
        .is_err()
    );
    assert_eq!(calls, 0);
    assert!(tiny.dictionary(&p, 1., 8., &mut work()).is_err());
}
#[test]
fn complete_original_constant_input_forecasts_leave_model_room_without_pruning() {
    let budget = Budget::new(PROCESS_BYTES, 1024 * 1024 * 1024).unwrap();
    for (states, classes, records, unknown) in
        [(100, 97, 4656, 5112868), (196, 193, 18528, 7324366)]
    {
        let sh = Shape {
            programs: 962806,
            states,
            classes,
        };
        let remaining = budget.after_input_forecast(sh, records, unknown).unwrap();
        let inputs = domain::dictionary_bytes(sh, sh.rows(), unknown).unwrap()
            + records::packed_bytes(sh, records).unwrap();
        assert_eq!(
            remaining as u128 + inputs + budget.external_bytes(),
            CORE_BYTES as u128
        );
        assert!(remaining > 50_000_000_000);
        assert_eq!(
            records::packed_bytes(sh, records).unwrap(),
            std::mem::size_of::<Packed>() as u128 + 2 * records as u128 * (962806 + 1)
        );
    }
    // These forecasts retain all original rows/candidates and measure only
    // their storage. A head, full update and observed RSS are still required.
}
