//! Real file and old-model controls for the prospective compact boundary.
use crate::*;
use joint::{
    Parameters, Shape,
    domain::{self, Domain},
    head::Head,
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
#[path = "tests/model.rs"]
mod model_tests;
#[path = "tests/probe.rs"]
mod probe_tests;
#[path = "tests/runner.rs"]
mod runner_tests;
static NEXT: AtomicU64 = AtomicU64::new(0);
const ROOM: u64 = 2 * 1024 * 1024 * 1024;
struct Fixture {
    root: PathBuf,
    guard: resources::Guard,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "compact-compare-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir(&root).unwrap();
        let guard = resources::Guard {
            workspace: root.clone(),
            storage: root.clone(),
            run: root.clone(),
            output_budget: 256 * 1024 * 1024,
            workspace_reserve: 0,
            storage_reserve: 0,
        };
        Self { root, guard }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}
fn point() -> (Domain, Parameters, Head, Work) {
    let shape = Shape {
        programs: 2,
        states: 8,
        classes: 3,
    };
    let mut w = work::endpoint(0).unwrap();
    let labels = (0..shape.rows())
        .map(|i| (i % 4 != 3).then_some(i % 3))
        .collect();
    let d = domain::from_labels(shape, labels, 1., 32., ROOM, &mut w).unwrap();
    let mut old = d.initial(8., &mut w).unwrap();
    let at = d.offsets[3];
    old.values[at..at + 3].copy_from_slice(&[-0., f64::from_bits(1), 4.]);
    let q = Parameters::from_legacy(&d, &old, ROOM, &mut w).unwrap();
    let h = Head::build(&d, &q, ROOM, &mut w).unwrap();
    (d, q, h, w)
}
#[test]
fn complete_checkpoint_keeps_every_word_and_accounts_for_read_and_head_storage() {
    let f = Fixture::new();
    let (d, q, h, mut w) = point();
    let path = f.root.join("model.bin");
    let saved = tensor::save(&path, &d, &q, &h, &f.guard, ROOM, &mut w).unwrap();
    assert_eq!(
        saved.artifact.bytes,
        tensor::length(d.shape, d.unknown.len()).unwrap()
    );
    let before = w.used;
    assert!(tensor::load(&path, &d, &saved.artifact, saved.body_sha256, 1, &mut w).is_err());
    assert_eq!(w.used, before);
    let (again, head) =
        tensor::load(&path, &d, &saved.artifact, saved.body_sha256, ROOM, &mut w).unwrap();
    assert_eq!(
        q.values().map(|x| x.to_bits()).collect::<Vec<_>>(),
        again.values().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    for p in 0..d.shape.programs {
        for s in 0..d.shape.states {
            for c in 0..d.shape.classes {
                let (a, b) = h.coefficient(p, s, c).unwrap();
                let (x, y) = head.coefficient(p, s, c).unwrap();
                assert_eq!((a.to_bits(), b.to_bits()), (x.to_bits(), y.to_bits()));
            }
        }
    }
    assert!(tensor::save(&path, &d, &q, &h, &f.guard, ROOM, &mut w).is_err());
    let original = std::fs::read(&path).unwrap();
    for at in [0, 24, 56, 56 + 4 * d.shape.rows(), original.len() - 1] {
        let mut bytes = original.clone();
        bytes[at] ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(
            tensor::load(
                &path,
                &d,
                &disk::hash(&path).unwrap(),
                saved.body_sha256,
                ROOM,
                &mut w
            )
            .is_err()
        );
    }
    std::fs::write(&path, original).unwrap();
    let mut labels = d.labels.clone();
    labels[0] = Some(1);
    let changed = domain::from_labels(d.shape, labels, 1., 32., ROOM, &mut w).unwrap();
    assert!(
        tensor::load(
            &path,
            &changed,
            &saved.artifact,
            saved.body_sha256,
            ROOM,
            &mut w
        )
        .unwrap_err()
        .to_string()
        .contains("original observed records")
    );
}
#[test]
fn two_required_mode_disk_forecasts_keep_the_complete_largest_logical_tensors() {
    let shape = Shape {
        programs: 962806,
        states: 196,
        classes: 193,
    };
    let f = resources::case_budget(shape, 18528, 18528, 7324366, 840996226).unwrap();
    assert_eq!(REQUIRED_MODES.len(), 2);
    assert_eq!(model::MODES, REQUIRED_MODES);
    assert_eq!(f.model_bytes, 2 * (88 + 4 * 188709976 + 8 * 1414565444));
    assert_eq!(
        f.total_bytes,
        f.model_bytes + f.raw_bytes + f.metadata_bytes
    );
    assert!(f.model_bytes > 24_000_000_000);
    assert!(resources::case_budget(shape, 18528, 0, 7324366, 840996226).is_err());
    assert!(tensor::length(shape, shape.rows() + 1).is_err());
    // Both original and repeat must fit each relevant filesystem.
    let a = resources::Volume {
        source: "root".into(),
        available: 100,
    };
    let b = resources::Volume {
        source: "storage".into(),
        available: 2 * f.total_bytes,
    };
    resources::admit(&a, &b, 100, 2 * f.total_bytes).unwrap();
    assert!(resources::admit(&a, &b, 101, 2 * f.total_bytes).is_err());
    assert!(resources::admit(&a, &b, 100, 2 * f.total_bytes + 1).is_err());
    assert!(resources::admit(&a, &a, 60, 50).is_err());
}
#[test]
fn local_memory_or_either_disk_shortage_stops_before_writes_and_checkpoint_work() {
    let f = Fixture::new();
    let (d, q, h, mut w) = point();
    let path = f.root.join("never.bin");
    let start = w.used;
    assert!(tensor::save(&path, &d, &q, &h, &f.guard, 1, &mut w).is_err());
    for workspace in [false, true] {
        let mut denied = f.guard.clone();
        if workspace {
            denied.workspace_reserve = u64::MAX;
        } else {
            denied.storage_reserve = u64::MAX;
        }
        assert!(tensor::save(&path, &d, &q, &h, &denied, ROOM, &mut w).is_err());
    }
    assert_eq!(w.used, start);
    assert!(!path.exists());
}
