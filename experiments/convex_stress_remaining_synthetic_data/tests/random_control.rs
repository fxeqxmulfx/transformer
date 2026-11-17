use convex_stress_remaining_synthetic_data::{Result, dataset, scoring};
use std::collections::BTreeSet;

#[test]
fn exact_generation_ceiling_for_one_visible_prompt() -> Result<()> {
    let data = dataset::build("random-lm-control")?;
    let expected = [
        ("train", 8, 1),
        ("train_clean", 8, 1),
        ("validation", 64, 1),
        ("validation/novel", 64, 1),
        ("test", 128, 2),
        ("validation/length-16", 64, 1),
        ("validation/length-16/novel", 64, 1),
        ("test/length-16", 128, 1),
        ("validation/length-32", 64, 1),
        ("validation/length-32/novel", 64, 1),
        ("test/length-32", 128, 1),
    ];
    assert_eq!(data.task.splits.len(), expected.len());
    for (split, count, expected_exact) in expected {
        let rows = &data.task.splits[split];
        assert_eq!(rows.len(), count, "{split}");
        let first = &rows[0];
        let mut answers = BTreeSet::new();
        for row in rows {
            assert_eq!(row.prompt, first.prompt, "{split}");
            assert_eq!(row.limit, first.limit, "{split}");
            assert_eq!(row.answer.len(), row.limit, "{split}");
            answers.insert(row.answer.as_slice());
        }

        // One deterministic rollout serves all rows. Every output absent from
        // this set scores zero exact matches, so the search is exhaustive.
        let mut max_exact = 0;
        for answer in answers {
            let output: Vec<usize> = answer
                .iter()
                .copied()
                .map(usize::try_from)
                .collect::<std::result::Result<_, _>>()?;
            let mut exact = 0;
            for row in rows {
                exact += usize::from(scoring::score(&row.answer, &output)?.exact);
            }
            max_exact = max_exact.max(exact);
        }
        assert_eq!(max_exact, expected_exact, "{split}");
    }
    Ok(())
}
