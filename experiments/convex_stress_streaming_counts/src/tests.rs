use crate::*;

mod index_tests;
mod stream_tests;

fn rows(length: u32) -> Vec<Observation> {
    let mut out = vec![];
    for n in 1..=length {
        for mut code in 0..3usize.pow(n) {
            let mut word = vec![1];
            for _ in 0..n {
                word.push([9, 10, 23][code % 3]);
                code /= 3;
            }
            let a = word.iter().filter(|&&x| x == 9).count();
            let b = word.iter().filter(|&&x| x == 10).count();
            out.push(Observation {
                id: out.len(),
                tokens: word,
                label: if a < b { 19 } else { 7 },
            });
        }
    }
    out
}
fn meter() -> latent::Meter {
    latent::Meter {
        work: 0,
        limit: search::WORK_LIMIT,
    }
}

#[test]
fn preflight_refuses_caps_and_preserves_the_original_shallow_result() {
    let rows = rows(2);
    let mut policy = search::Policy::default();
    let (work, bytes) = search::preflight(&rows, 32, &policy).unwrap();
    assert_eq!(work, search::WORK_LIMIT);
    assert!(u128::from(bytes) < prior::joint::MAX_BYTES);
    policy.cache_entries = 65537;
    assert!(search::preflight(&rows, 32, &policy).is_err());
    policy.cache_entries = 65536;
    policy.quotient.latent_classes = usize::MAX;
    assert!(search::preflight(&rows, 32, &policy).is_err());
    let report = search::run(&rows, 32, search::Policy::default()).unwrap();
    let old_policy = previous::search::Policy {
        depth: 2,
        ..Default::default()
    };
    let old = previous::search::run(&rows, 32, old_policy).unwrap();
    assert_eq!(report.shallow, old);
    assert!(!report.programs.is_empty() && report.deep.is_none());
    assert_eq!(report.programs, old.programs);
}

#[test]
fn shallowly_truncated_banks_still_allow_audited_streaming_fits() {
    let rows = rows(3);
    let pool = pool::Pool::new(&rows).unwrap();
    let children = bank::Bank::initial(&pool, 8).unwrap();
    let mut policy = search::Policy::default();
    policy.quotient.latent_classes = 8;
    policy.cache_entries = 2;
    let report = deep::run(&pool, &rows, &children, &policy, 0).unwrap();
    assert!(!report.programs.is_empty(), "{:?}", report.refused);
    assert!(
        report
            .attempts
            .iter()
            .any(|a| a.left.capped || a.right.as_ref().is_some_and(|s| s.capped))
    );
    for program in report.programs {
        let mut labels = std::collections::BTreeMap::new();
        for r in &rows {
            let atom = program.interpret(&r.tokens).unwrap();
            assert_eq!(program.run(&r.tokens), Some(atom));
            if let Some(old) = labels.insert(atom, r.label) {
                assert_eq!(old, r.label);
            }
        }
    }
}
