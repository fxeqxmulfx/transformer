use super::*;
fn training() -> Training {
    let mut rows = Vec::with_capacity(16);
    rows.extend([[0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0]]);
    Training {
        vocabulary: 2,
        rows,
    }
}
#[test]
fn complete_family_materialization_matches_the_prior_words_dictionary_and_total_work() {
    let t = training();
    let expected = pairs::selection::select(&t, &mut Work::new(pairs::MAX_WORK).unwrap()).unwrap();
    assert_eq!(expected.retained.len(), pairs::PROGRAMS);
    let mut phases = Phases::new().unwrap();
    let new = prepare(
        &t,
        &expected,
        4 * 1024u64.pow(3),
        16 * 1024 * 1024,
        &mut phases,
    )
    .unwrap();
    assert_eq!(new.selection, expected);
    assert_eq!(phases.selection.used, expected.operations);
    let mut old_work = Work::new(pairs::MAX_WORK).unwrap();
    let (old_domain, old_records) =
        pairs::encoding::encode_packed(&t, &expected, &mut old_work).unwrap();
    assert_eq!(old_domain, new.domain);
    assert_eq!(old_work.used, phases.total().unwrap());
    assert_eq!(new.records.len(), 4);
    for r in 0..4 {
        assert_eq!(old_records.label(r).unwrap(), new.records.label(r).unwrap());
        for p in 0..pairs::PROGRAMS {
            assert_eq!(
                old_records.state(r, p).unwrap(),
                new.records.state(r, p).unwrap()
            );
        }
    }
    assert!(new.budget.model_room(&new.domain, &new.records).unwrap() > 1024 * 1024 * 1024);
    let mut visited = 0;
    joint::records::Source::Packed(&new.records)
        .visit(3..4, &mut phases.materialization, |row, _| {
            assert_eq!(row.states.len(), pairs::PROGRAMS);
            assert_eq!(row.label, 0);
            visited += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(visited, 1);
}
#[test]
fn complete_selection_and_caller_capacity_are_checked_before_materialization() {
    let t = training();
    let expected = pairs::selection::select(&t, &mut Work::new(pairs::MAX_WORK).unwrap()).unwrap();
    let mut phases = Phases::new().unwrap();
    assert!(
        prepare(
            &t,
            &expected,
            joint::admission::MIN_RUNTIME_BYTES + 64 * 1024 * 1024,
            0,
            &mut phases
        )
        .is_err()
    );
    assert_eq!(phases.total().unwrap(), 0);
    let mut missing = expected;
    missing.retained.pop();
    let error = prepare(&t, &missing, 4 * 1024u64.pow(3), 0, &mut phases)
        .err()
        .unwrap();
    assert!(error.to_string().contains("omits or changes"));
    assert!(phases.selection.used > 0);
    assert_eq!(phases.materialization.used, 0);
    assert!(prepare(&t, &missing, 4 * 1024u64.pow(3), 0, &mut phases).is_err());
}
#[test]
fn separate_bounded_passes_keep_the_complete_total_and_reject_corrupt_counters() {
    let mut phases = Phases::new().unwrap();
    phases
        .selection
        .charge(pairs::MAX_WORK as u128 - 1)
        .unwrap();
    assert!(phases.selection.charge(2).is_err());
    phases.materialization.charge(17).unwrap();
    assert_eq!(phases.total().unwrap(), pairs::MAX_WORK + 16);
    phases.materialization.limit += 1;
    assert!(phases.total().is_err());
    phases.materialization.limit = pairs::MAX_WORK;
    phases.materialization.used = pairs::MAX_WORK + 1;
    assert!(phases.total().is_err());
}
#[test]
fn transient_reserves_cover_the_pinned_native_grammar_and_interpreter_tables() {
    let bank = programs();
    assert_eq!(bank.len(), pairs::PROGRAMS);
    // Each enum has only fixed-width fields; reserve space for the full bank,
    // serialized bank growth, both partition vectors and their moving growth.
    let bank_storage = memory::vector(&bank);
    let partition =
        3 * pairs::PROGRAMS as u128 * std::mem::size_of::<pairs::selection::Conflict>() as u128;
    let encoded = serde_json::to_vec(&bank).unwrap();
    let serialized = 3 * memory::vector(&encoded);
    assert!(bank_storage + partition + serialized + 16 * 1024 * 1024 < memory::SELECTION_SCRATCH);
    assert!(
        2 * 1024 * 7 * std::mem::size_of::<usize>() as u128
            + std::mem::size_of::<Evaluator>() as u128
            + 65536
            < memory::INTERPRETER_SCRATCH
    );
    assert!(memory::training(&training()) > std::mem::size_of::<Training>() as u128 + 4 * 12);
}
