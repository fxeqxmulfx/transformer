use super::*;
use crate::head::Head;
const LIMIT: u64 = 16 * 1024 * 1024;
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}
fn fixture() -> (Domain, legacy::Parameters, Parameters) {
    let d = Domain::from_labels(
        Shape {
            programs: 2,
            states: 8,
            classes: 5,
        },
        vec![
            Some(0),
            Some(1),
            None,
            None,
            Some(2),
            Some(3),
            Some(4),
            Some(4),
            Some(1),
            None,
            Some(0),
            None,
            Some(4),
            Some(4),
            Some(3),
            Some(2),
        ],
        1.,
        8.,
        &mut work(),
    )
    .unwrap();
    let mut q = d.initial(4., &mut work()).unwrap();
    let at = d.offsets[3];
    q.values[at..at + 5].copy_from_slice(&[2., -0., f64::from_bits(1), 0., 0.]);
    let compact = Parameters::from_legacy(&d, &q, LIMIT, &mut work()).unwrap();
    (d, q, compact)
}
fn rehash(bytes: &mut [u8]) -> [u8; 32] {
    let end = bytes.len() - 32;
    let hash: [u8; 32] = Sha256::digest(&bytes[..end]).into();
    bytes[end..].copy_from_slice(&hash);
    hash
}
#[test]
fn logical_stream_is_byte_identical_and_both_readers_reconstruct_every_word() {
    let (d, q, compact) = fixture();
    let (mut wa, mut wb) = (work(), work());
    let (mut old, mut new) = (vec![], vec![]);
    let a = legacy::persistence::write(&d, &q, &mut old, &mut wa).unwrap();
    let b = write(&d, &compact, &mut new, &mut wb).unwrap();
    assert_eq!(a, b);
    assert_eq!(old, new);
    assert_eq!(wa.used, wb.used);
    assert_eq!(bytes(&d), new.len() as u64);
    let (ld, lq) = legacy::persistence::read(new.as_slice(), b, new.len() as u64, &mut wa).unwrap();
    let (cd, cq) = read(old.as_slice(), a, old.len() as u64, LIMIT, &mut wb).unwrap();
    assert_eq!(ld, cd);
    assert_eq!(d, cd);
    assert_eq!(wa.used, wb.used);
    assert_eq!(bits(q.values.iter().copied()), bits(cq.values().copied()));
    assert_eq!(bits(lq.values), bits(cq.values().copied()));
    let before = Head::build(&d, &compact, LIMIT, &mut work()).unwrap();
    let after = Head::build(&cd, &cq, LIMIT, &mut work()).unwrap();
    for p in 0..2 {
        for f in 0..8 {
            for c in 0..5 {
                let a = before.coefficient(p, f, c).unwrap();
                let b = after.coefficient(p, f, c).unwrap();
                assert_eq!(
                    (a.0.to_bits(), a.1.to_bits()),
                    (b.0.to_bits(), b.1.to_bits())
                );
            }
        }
    }
    for a in 0..8 {
        for b in 0..8 {
            assert_eq!(
                bits(before.logits(&[a, b], &mut work()).unwrap()),
                bits(after.logits(&[a, b], &mut work()).unwrap())
            );
        }
    }
}

#[test]
fn compact_read_keeps_full_wire_data_and_refuses_oversized_nonuniform_fallback() {
    let sh = Shape {
        programs: 1,
        states: 128,
        classes: 1024,
    };
    let d = Domain::from_labels(sh, vec![None; 128], 1., 8., &mut work()).unwrap();
    let q = Parameters::initial(&d, 8., LIMIT, &mut work()).unwrap();
    let mut stream = vec![];
    let hash = write(&d, &q, &mut stream, &mut work()).unwrap();
    assert!(stream.len() > 1_000_000);
    let (_, loaded) = read(
        stream.as_slice(),
        hash,
        stream.len() as u64,
        24_576,
        &mut work(),
    )
    .unwrap();
    assert!(loaded.allocation_bytes() < 8192);
    assert_eq!(bits(q.values().copied()), bits(loaded.values().copied()));
    let mut full = d.initial(8., &mut work()).unwrap();
    for values in full.values[1..].chunks_mut(1024) {
        for (c, x) in values.iter_mut().enumerate() {
            *x = 16. * (c + 1) as f64 / (1024. * 1025.);
        }
    }
    stream.clear();
    let hash = legacy::persistence::write(&d, &full, &mut stream, &mut work()).unwrap();
    let error = read(
        stream.as_slice(),
        hash,
        stream.len() as u64,
        24_576,
        &mut work(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("storage"));
    let (_, loaded) = read(
        stream.as_slice(),
        hash,
        stream.len() as u64,
        LIMIT,
        &mut work(),
    )
    .unwrap();
    assert_eq!(loaded.unknown_rows().dense_rows(), 128);
    assert_eq!(bits(full.values), bits(loaded.values().copied()));
}

#[test]
fn corrupt_counts_coordinates_hashes_and_even_rehashed_infeasibility_are_rejected() {
    let (d, _, q) = fixture();
    let mut stream = vec![];
    let hash = write(&d, &q, &mut stream, &mut work()).unwrap();
    for pos in [
        0,
        8,
        16,
        24,
        32,
        40,
        56,
        56 + 4 * d.shape.rows(),
        stream.len() - 1,
    ] {
        let mut bad = stream.clone();
        bad[pos] ^= 1;
        assert!(read(bad.as_slice(), hash, bad.len() as u64, LIMIT, &mut work()).is_err());
    }
    for value in [f64::NAN, f64::INFINITY, -1., 1e13] {
        let mut bad = stream.clone();
        let at = 56 + 4 * d.shape.rows();
        bad[at..at + 8].copy_from_slice(&value.to_le_bytes());
        let updated = rehash(&mut bad);
        assert!(
            read(
                bad.as_slice(),
                updated,
                bad.len() as u64,
                LIMIT,
                &mut work()
            )
            .is_err()
        );
    }
    let mut bad = stream.clone();
    bad[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    let updated = rehash(&mut bad);
    assert!(
        read(
            bad.as_slice(),
            updated,
            bad.len() as u64,
            LIMIT,
            &mut work()
        )
        .is_err()
    );
    let mut bad = stream[..56 + 4 * d.shape.rows() + 8 * d.shape.programs].to_vec();
    bad[32..40].copy_from_slice(&0u64.to_le_bytes());
    bad.extend([0; 32]);
    let updated = rehash(&mut bad);
    let error = read(
        bad.as_slice(),
        updated,
        bad.len() as u64,
        LIMIT,
        &mut work(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("before allocation"));
    let mut extra = stream.clone();
    extra.push(0);
    assert!(
        read(
            extra.as_slice(),
            hash,
            extra.len() as u64,
            LIMIT,
            &mut work()
        )
        .is_err()
    );
    assert!(
        read(
            extra.as_slice(),
            hash,
            stream.len() as u64,
            LIMIT,
            &mut work()
        )
        .is_err()
    );
    assert!(
        read(
            &stream[..stream.len() - 1],
            hash,
            stream.len() as u64,
            LIMIT,
            &mut work()
        )
        .is_err()
    );
    assert!(read(stream.as_slice(), hash, stream.len() as u64, 1, &mut work()).is_err());
    assert!(
        read(
            stream.as_slice(),
            hash,
            stream.len() as u64,
            LIMIT,
            &mut Work::new(1).unwrap()
        )
        .is_err()
    );
}

#[test]
fn guarded_writes_check_both_disks_keep_repeat_reserve_and_never_overwrite() {
    use legacy::resources::sufficient;
    let g = 1024u64.pow(3);
    assert!(sufficient(100 * g, 16 * g - 1, false, 0).is_err());
    assert!(sufficient(2 * g - 1, 100 * g, false, 0).is_err());
    assert!(sufficient(18 * g, 18 * g, true, 1).is_err());
    assert!(sufficient(18 * g + 1, 18 * g + 1, true, 1).is_ok());
    assert!(sufficient(u64::MAX, u64::MAX, true, u64::MAX).is_err());
    let (d, _, q) = fixture();
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "compact-checkpoint-{}-{suffix}",
        std::process::id()
    ));
    std::fs::create_dir(&root).unwrap();
    let path = root.join("checkpoint.bin");
    assert!(write_file(&d, &q, &root, &root, &path, u64::MAX, &mut work()).is_err());
    assert!(!path.exists());
    let hash = write_file(&d, &q, &root, &root, &path, 0, &mut work()).unwrap();
    let saved = std::fs::read(&path).unwrap();
    read(
        saved.as_slice(),
        hash,
        saved.len() as u64,
        LIMIT,
        &mut work(),
    )
    .unwrap();
    assert!(write_file(&d, &q, &root, &root, &path, 0, &mut work()).is_err());
    assert_eq!(saved, std::fs::read(&path).unwrap());
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(root).unwrap();
}

struct FailingWriter {
    remaining: usize,
    fail_flush: bool,
}
impl Write for FailingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Err(std::io::Error::other("simulated storage failure"));
        }
        let n = buf.len().min(self.remaining);
        self.remaining -= n;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if self.fail_flush {
            Err(std::io::Error::other("simulated flush failure"))
        } else {
            Ok(())
        }
    }
}
#[test]
fn streaming_storage_and_flush_errors_propagate_without_success_receipts() {
    let (d, _, q) = fixture();
    for remaining in [0, 7, 57, bytes(&d) as usize - 1] {
        assert!(
            write(
                &d,
                &q,
                FailingWriter {
                    remaining,
                    fail_flush: false
                },
                &mut work()
            )
            .is_err()
        );
    }
    assert!(
        write(
            &d,
            &q,
            FailingWriter {
                remaining: bytes(&d) as usize,
                fail_flush: true
            },
            &mut work()
        )
        .is_err()
    );
}
