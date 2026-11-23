#[path = "../src/stress/recall_gap.rs"]
mod recall_gap;

use basis::{generator, wire};
use std::collections::BTreeSet;

#[test]
fn zero_gap_reproduces_the_frozen_easy_test() -> basis::Result<()> {
    let name = "basis-recall-easy";
    let spec = recall_gap::source_spec(name)?;
    let rows = generator::generate(&spec, "test", 512, &BTreeSet::new(), false)?;
    let (digest, paired) = recall_gap::delay(name, &rows, 0)?;
    let original: serde_json::Value = serde_json::from_str(arithmetic::ORIGINAL35)?;
    let frozen = original["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == name)
        .unwrap();
    assert_eq!(frozen["data_fingerprints"]["test"], digest);
    assert!(
        rows.iter()
            .zip(&paired)
            .all(|(a, b)| wire::value(a) == wire::value(b))
    );
    Ok(())
}

#[test]
fn longer_gaps_preserve_each_write_query_and_shorter_filler_stream() -> basis::Result<()> {
    for name in ["basis-recall-easy", "basis-recall-hard"] {
        let spec = recall_gap::source_spec(name)?;
        let mut source = generator::generate(&spec, "test", 20, &BTreeSet::new(), false)?;
        let (first_hash, first) = recall_gap::delay(name, &source, 7)?;
        let (next_hash, next) = recall_gap::delay(name, &source, 37)?;
        assert_ne!(first_hash, next_hash);
        let table_end = 1 + 2 * generator::number(&spec, "pairs")?;
        let limit = 8192 - generator::number(&spec, "length")?;
        let (_, longest) = recall_gap::delay(name, &source[..1], limit)?;
        assert_eq!(
            &source[0].tokens[table_end..],
            &longest[0].tokens[table_end + limit..]
        );
        assert_eq!(
            &source[0].targets[table_end..],
            &longest[0].targets[table_end + limit..]
        );
        assert!(recall_gap::delay(name, &source[..1], limit + 1).is_err());
        for ((a, b), c) in source.iter().zip(first).zip(next) {
            assert_eq!(&a.tokens[..table_end], &c.tokens[..table_end]);
            assert_eq!(
                &b.tokens[table_end..table_end + 7],
                &c.tokens[table_end..table_end + 7]
            );
            assert_eq!(&a.tokens[table_end..], &c.tokens[table_end + 37..]);
            assert_eq!(&a.targets[table_end..], &c.targets[table_end + 37..]);
            assert_eq!(&a.changes[table_end..], &c.changes[table_end + 37..]);
            assert!(
                c.targets[table_end..table_end + 37]
                    .iter()
                    .all(|&t| t == -100)
            );
            assert_eq!(c.targets.iter().filter(|&&y| y >= 0).count(), 8);
        }
        assert!(recall_gap::delay(name, &source, usize::MAX).is_err());
        source[0].tokens[1] = 0;
        assert!(recall_gap::delay(name, &source, 7).is_err());
    }
    assert!(recall_gap::source_spec("basis-parity16").is_err());
    Ok(())
}
