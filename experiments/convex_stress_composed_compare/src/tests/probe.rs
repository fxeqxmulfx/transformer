//! Complete report readback, strict metadata, original candidate partition,
//! byte limits and manifest bindings. No original heldout data is read.
use super::*;
use crate::runner::{self, probe::Report};
use std::collections::BTreeMap;

fn report() -> Report {
    let input = pairs::Training {
        vocabulary: 3,
        rows: vec![[0, 0, 0], [0, 1, 1], [1, 0, 2], [0, 0, 0]],
    };
    let mut work = Work::new(crate::MAX_WORK).unwrap();
    let selection = pairs::selection::select(&input, &mut work).unwrap();
    Report {
        selection,
        work,
        protocol_sha256: pairs::digest(pairs::PROTOCOL.as_bytes()),
        expanded_training_records_allocated: 0,
        heldout_pools_read: 0,
        accuracy_measurements: 0,
        training_updates: 0,
    }
}

#[test]
fn typed_probe_keeps_every_candidate_above_the_old_json_limit_and_checks_bound_hashes() {
    let fixture = Fixture::new();
    let original = report();
    original.validate().unwrap();
    let path = fixture.root.join("inputs/probe/synthetic.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = std::fs::File::create(&path).unwrap();
    serde_json::to_writer(&mut file, &original).unwrap();
    let length = file.metadata().unwrap().len();
    // Valid trailing whitespace crosses the previous parser's size cap,
    // while the complete semantic record and all candidate IDs stay intact.
    let extra = (runner::JSON_LIMIT + 1).saturating_sub(length);
    std::io::copy(
        &mut std::io::Read::take(std::io::repeat(b' '), extra),
        &mut file,
    )
    .unwrap();
    file.flush().unwrap();
    assert!(file.metadata().unwrap().len() > runner::JSON_LIMIT);
    assert!(file.metadata().unwrap().len() <= runner::PROBE_LIMIT);
    assert_eq!(runner::probe::read(&path).unwrap(), original.selection);
    assert!(runner::read::<Report>(&path).is_err());
    let mut inventory = runner::Inventory {
        protocol: runner::protocol().unwrap(),
        producer_commit: "0".repeat(40),
        cases: vec![],
        inputs: BTreeMap::new(),
        single_run_budget: 1,
    };
    inventory.inputs.insert(
        "inputs/probe/synthetic.json".into(),
        disk::hash(&path).unwrap(),
    );
    assert_eq!(
        runner::selection(&fixture.root, &inventory, "synthetic").unwrap(),
        original.selection
    );
    inventory
        .inputs
        .get_mut("inputs/probe/synthetic.json")
        .unwrap()
        .sha256 = "0".repeat(64);
    assert!(runner::selection(&fixture.root, &inventory, "synthetic").is_err());
    file.set_len(runner::PROBE_LIMIT + 1).unwrap();
    assert!(
        runner::probe::read(&path)
            .unwrap_err()
            .to_string()
            .contains("128 MiB")
    );
}

#[test]
fn typed_probe_rejects_tainted_metadata_and_a_complete_length_but_forged_partition() {
    let original = report();
    assert!(!original.selection.retained.is_empty());
    assert!(!original.selection.conflicts.is_empty());
    for variant in 0..12 {
        let mut changed = original.clone();
        match variant {
            0 => changed.heldout_pools_read = 1,
            1 => changed.accuracy_measurements = 1,
            2 => changed.training_updates = 1,
            3 => changed.expanded_training_records_allocated = 1,
            4 => changed.protocol_sha256 = "0".repeat(64),
            5 => changed.work.limit -= 1,
            6 => changed.work.used -= 1,
            7 => changed.selection.retained[0] = changed.selection.conflicts[0].program,
            8 => changed.selection.conflicts[0].second_record = changed.selection.records,
            9 => changed.selection.conflicts[0].state = changed.selection.vocabulary,
            10 => changed.selection.audit.independently_checked_states -= 1,
            11 => changed.selection.audit.unknown_rows += 1,
            _ => unreachable!(),
        }
        assert!(changed.validate().is_err(), "tampering variant {variant}");
    }
    let fixture = Fixture::new();
    let path = fixture.root.join("tainted.json");
    let mut changed = original.clone();
    changed.training_updates = 1;
    serde_json::to_writer(std::fs::File::create(&path).unwrap(), &changed).unwrap();
    assert!(runner::probe::read(&path).is_err());
    let bytes = serde_json::to_vec(&original).unwrap();
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&bytes[..bytes.len() - 1]).unwrap();
    file.write_all(b",\"heldout\":[[1,1,2]]}").unwrap();
    file.flush().unwrap();
    assert!(runner::probe::read(&path).is_err());
}
