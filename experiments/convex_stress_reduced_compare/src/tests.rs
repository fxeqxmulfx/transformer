//! Boundary regressions adapted from bounded_compare ca6a4977. These check
//! free-coordinate persistence, complete-head reconstruction and actual I/O.
use crate::{
    disk,
    resources::{self, Guard, Volume},
    tensor, *,
};
use joint::{Parameters, Shape, Work, domain::Domain, head::Head};
use std::{
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
mod model;
struct Fixture {
    root: PathBuf,
    guard: Guard,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "reduced-comparison-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir(&root).unwrap();
        let guard = Guard {
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
    let mut work = Work::new(crate::MAX_WORK).unwrap();
    let labels = (0..shape.rows())
        .map(|i| (i % 4 != 3).then_some(i % 3))
        .collect();
    let domain = Domain::from_labels(shape, labels, 1., 32., &mut work).unwrap();
    let mut q = domain.initial(8., &mut work).unwrap();
    let start = domain.offsets[3];
    q.values[start..start + 3].copy_from_slice(&[-0., f64::from_bits(1), 4.]);
    let head = Head::build(&domain, &q, &mut work).unwrap();
    (domain, q, head, work)
}
#[test]
fn complete_free_tensor_and_reconstructed_complex_head_preserve_bits() {
    let fixture = Fixture::new();
    let (domain, q, h, mut work) = point();
    let path = fixture.root.join("model.bin");
    let saved = tensor::save(&path, &domain, &q, &h, &fixture.guard, &mut work).unwrap();
    assert_eq!(
        saved.artifact.bytes,
        tensor::length(h.shape, domain.unknown.len()).unwrap()
    );
    assert_eq!(saved.artifact, disk::hash(&path).unwrap());
    let (again, head) = tensor::load(
        &path,
        &domain,
        &saved.artifact,
        saved.body_sha256,
        &mut work,
    )
    .unwrap();
    for (a, b) in q.values.iter().zip(&again.values) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for p in 0..h.shape.programs {
        for s in 0..h.shape.states {
            for c in 0..h.shape.classes {
                let (ar, ai) = h.coefficient(p, s, c).unwrap();
                let (br, bi) = head.coefficient(p, s, c).unwrap();
                assert_eq!((ar.to_bits(), ai.to_bits()), (br.to_bits(), bi.to_bits()));
            }
        }
    }
    assert!(tensor::save(&path, &domain, &q, &h, &fixture.guard, &mut work).is_err());
    let copy = fixture.root.join("copy.bin");
    assert_eq!(
        disk::copy(&path, &copy, &saved.artifact, &fixture.guard).unwrap(),
        saved.artifact
    );
}
#[test]
fn changed_headers_coordinates_and_dictionaries_fail_even_with_new_file_hashes() {
    let fixture = Fixture::new();
    let (domain, q, h, mut work) = point();
    let path = fixture.root.join("model.bin");
    let saved = tensor::save(&path, &domain, &q, &h, &fixture.guard, &mut work).unwrap();
    let original = std::fs::read(&path).unwrap();
    for position in [0, 24, 56, 56 + 4 * domain.shape.rows(), original.len() - 1] {
        let mut bytes = original.clone();
        bytes[position] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            tensor::load(
                &path,
                &domain,
                &saved.artifact,
                saved.body_sha256,
                &mut work
            )
            .is_err()
        );
        assert!(
            tensor::load(
                &path,
                &domain,
                &disk::hash(&path).unwrap(),
                saved.body_sha256,
                &mut work
            )
            .is_err()
        );
    }
    let mut bytes = original.clone();
    bytes.push(0);
    std::fs::write(&path, bytes).unwrap();
    assert!(
        tensor::load(
            &path,
            &domain,
            &disk::hash(&path).unwrap(),
            saved.body_sha256,
            &mut work
        )
        .is_err()
    );
    std::fs::write(&path, original).unwrap();
    let mut labels = domain.labels.clone();
    labels[0] = Some(1);
    let changed = Domain::from_labels(domain.shape, labels, 1., 32., &mut work).unwrap();
    assert!(
        tensor::load(
            &path,
            &changed,
            &saved.artifact,
            saved.body_sha256,
            &mut work
        )
        .unwrap_err()
        .to_string()
        .contains("original observed records")
    );
}
#[test]
fn both_disks_and_shared_filesystem_capacity_are_checked() {
    let a = Volume {
        source: "root".into(),
        available: 100,
    };
    let b = Volume {
        source: "storage".into(),
        available: 200,
    };
    resources::admit(&a, &b, 100, 200).unwrap();
    assert!(resources::admit(&a, &b, 101, 1).is_err());
    assert!(resources::admit(&a, &b, 1, 201).is_err());
    assert!(resources::admit(&a, &a, 60, 50).is_err());
    resources::admit(&a, &a, 40, 60).unwrap();
    assert!(resources::admit(&a, &a, u64::MAX, 1).is_err());
}
#[test]
fn disk_failure_stops_before_checkpoint_work_and_never_overwrites() {
    let fixture = Fixture::new();
    let path = fixture.root.join("small.bin");
    let mut writer = disk::Writer::new(&path, 4, &fixture.guard).unwrap();
    writer.write_all(b"1234").unwrap();
    assert!(writer.write_all(b"5").is_err());
    assert_eq!(writer.finish().unwrap().bytes, 4);
    let mut denied = fixture.guard.clone();
    denied.workspace_reserve = u64::MAX;
    let (domain, q, h, mut work) = point();
    let before = work.used;
    let never = fixture.root.join("never.bin");
    assert!(tensor::save(&never, &domain, &q, &h, &denied, &mut work).is_err());
    assert_eq!(work.used, before);
    assert!(!never.exists());
    denied = fixture.guard.clone();
    denied.storage_reserve = u64::MAX;
    assert!(disk::Writer::new(&never, 4, &denied).is_err());
    assert!(!never.exists());
    assert!(disk::Writer::new(&path, 4, &fixture.guard).is_err());
    assert!(disk::Writer::new(&fixture.root.join("../escaped.bin"), 4, &fixture.guard).is_err());
    let mut over = fixture.guard.clone();
    over.output_budget = 3;
    assert!(over.check().is_err());
}
#[test]
fn compact_forecasts_keep_every_unknown_class_and_all_three_modes() {
    let shape = Shape {
        programs: 4767,
        states: 196,
        classes: 193,
    };
    let unknown = 3 * shape.programs;
    let columns = shape.programs * (48 * 4 + 193);
    let forecast = resources::case_budget(shape, 18528, 18528, unknown, columns).unwrap();
    assert_eq!(
        forecast.model_bytes,
        3 * tensor::length(shape, unknown).unwrap()
    );
    assert!((60_000_000..100_000_000).contains(&forecast.model_bytes));
    assert!(forecast.raw_bytes > 1_000_000_000);
    assert_eq!(
        forecast.total_bytes,
        forecast.model_bytes + forecast.raw_bytes + forecast.metadata_bytes
    );
    assert!(resources::case_budget(shape, 18528, 0, unknown, columns).is_err());
    assert!(
        resources::case_budget(
            Shape {
                programs: 299551,
                states: 196,
                classes: 193
            },
            18528,
            18528,
            299551 * 196,
            299551 * 49 * 193
        )
        .is_err()
    );
    assert!(tensor::length(shape, shape.rows() + 1).is_err());
    assert_eq!(RAM_LIMIT_BYTES, 110_000_000_000);
}
