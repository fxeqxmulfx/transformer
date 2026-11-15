use crate::{
    disk,
    resources::{self, Guard, Volume},
    tensor, *,
};
use joint::{Parameters, Shape, Work, head::Head};
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
            "bounded-artifact-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir(&root).unwrap();
        let guard = Guard {
            workspace: root.clone(),
            storage: root.clone(),
            run: root.clone(),
            output_budget: 1024 * 1024,
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
fn point() -> (Parameters, Head, Work) {
    let shape = Shape {
        programs: 2,
        states: 8,
        classes: 3,
    };
    let mut work = Work::new(joint::MAX_WORK).unwrap();
    let mut q = Parameters {
        values: (0..shape.parameters())
            .map(|i| ((i * 17) % 31) as f64 / 16.)
            .collect(),
    };
    q.values[shape.at(0, 1, 0)] = -0.;
    q.values[shape.at(1, 7, 2)] = f64::from_bits(1);
    let head = Head::build(shape, &q, &mut work).unwrap();
    (q, head, work)
}
#[test]
fn complete_tensor_roundtrip_preserves_every_binary64_bit() {
    let fixture = Fixture::new();
    let (q, h, mut work) = point();
    let path = fixture.root.join("model.bin");
    let saved = tensor::save(&path, &q, &h, &fixture.guard, &mut work).unwrap();
    assert_eq!(saved.bytes, tensor::length(h.shape).unwrap());
    assert_eq!(saved, disk::hash(&path).unwrap());
    let (again, head) = tensor::load(&path, h.shape, &saved, &mut work).unwrap();
    for (a, b) in q
        .values
        .iter()
        .chain(&h.real)
        .chain(&h.imaginary)
        .zip(again.values.iter().chain(&head.real).chain(&head.imaginary))
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(tensor::save(&path, &q, &h, &fixture.guard, &mut work).is_err());
    let copy = fixture.root.join("copy.bin");
    assert_eq!(
        disk::copy(&path, &copy, &saved, &fixture.guard).unwrap(),
        saved
    );
}
#[test]
fn rejects_changed_header_trailing_data_and_rehashed_wrong_coefficients() {
    let fixture = Fixture::new();
    let (q, h, mut work) = point();
    let path = fixture.root.join("model.bin");
    let saved = tensor::save(&path, &q, &h, &fixture.guard, &mut work).unwrap();
    let original = std::fs::read(&path).unwrap();
    let mut bytes = original.clone();
    bytes[0] = b'X';
    std::fs::write(&path, &bytes).unwrap();
    assert!(
        tensor::load(&path, h.shape, &saved, &mut work)
            .unwrap_err()
            .to_string()
            .contains("hash")
    );
    assert!(
        tensor::load(&path, h.shape, &disk::hash(&path).unwrap(), &mut work)
            .unwrap_err()
            .to_string()
            .contains("encoding")
    );
    let mut bytes = original.clone();
    bytes.extend_from_slice(&[0; 8]);
    std::fs::write(&path, &bytes).unwrap();
    assert!(tensor::load(&path, h.shape, &disk::hash(&path).unwrap(), &mut work).is_err());
    let mut bytes = original.clone();
    let i = 32 + 8 * q.values.len();
    bytes[i..i + 8].copy_from_slice(&1.234f64.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(
        tensor::load(&path, h.shape, &disk::hash(&path).unwrap(), &mut work)
            .unwrap_err()
            .to_string()
            .contains("joint tensor")
    );
    let mut bytes = original;
    bytes[32..40].copy_from_slice(&f64::NAN.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(tensor::load(&path, h.shape, &disk::hash(&path).unwrap(), &mut work).is_err());
}
#[test]
fn both_disk_admission_stops_on_either_shortage_and_sums_a_shared_filesystem() {
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
fn writer_stops_before_exceeding_budget_and_does_not_overwrite() {
    let fixture = Fixture::new();
    let path = fixture.root.join("small.bin");
    let mut writer = disk::Writer::new(&path, 4, &fixture.guard).unwrap();
    writer.write_all(b"1234").unwrap();
    assert!(writer.write_all(b"5").is_err());
    let saved = writer.finish().unwrap();
    assert_eq!(saved.bytes, 4);
    assert_eq!(std::fs::read(&path).unwrap(), b"1234");
    let mut denied = fixture.guard.clone();
    denied.workspace_reserve = u64::MAX;
    let never = fixture.root.join("never.bin");
    assert!(disk::Writer::new(&never, 4, &denied).is_err());
    assert!(!never.exists());
    assert!(disk::Writer::new(&fixture.root.join("../escaped.bin"), 4, &fixture.guard).is_err());
    let mut over = fixture.guard.clone();
    over.output_budget = 3;
    assert!(over.check().is_err());
}
#[test]
fn complete_original_and_repeat_forecasts_count_all_classes_and_modes() {
    let shape = Shape {
        programs: 4767,
        states: 196,
        classes: 193,
    };
    let forecast = resources::case_budget(shape, 18528, 18528).unwrap();
    assert_eq!(forecast.model_bytes, 3 * tensor::length(shape).unwrap());
    assert!(forecast.model_bytes > 12_000_000_000);
    assert!(forecast.raw_bytes > 1_000_000_000);
    assert_eq!(
        forecast.total_bytes,
        forecast.model_bytes + forecast.raw_bytes + forecast.metadata_bytes
    );
    assert!(2 * forecast.total_bytes < 60_000_000_000);
    assert!(resources::case_budget(shape, 18528, 0).is_err());
    assert!(
        tensor::length(Shape {
            programs: 65536,
            states: 4096,
            classes: 1024
        })
        .is_err()
    );
    assert_eq!(RAM_LIMIT_BYTES, 110_000_000_000);
}
