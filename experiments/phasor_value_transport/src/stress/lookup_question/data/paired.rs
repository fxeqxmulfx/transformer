//! Held-out paired prompts for testing the model’s use of the printed tables.
use super::*;

pub(in crate::stress::lookup_question) struct Paired {
    pub letters: [i64; 5],
    pub groups: BTreeMap<usize, [Vec<Example>; 4]>,
    pub digest: String,
}

/// Intervene after holding out each entire question. Cycling the first table's
/// right column preserves every token count and changes the oracle answer.
pub(in crate::stress::lookup_question) fn paired(original: &Task) -> Result<Paired> {
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as i64))
        .collect();
    let mut seen: HashSet<_> = original
        .splits
        .values()
        .flatten()
        .map(|e| e.prompt.clone())
        .collect();
    let mut groups = BTreeMap::new();
    for hops in [1, 2, 3, 5, 10] {
        let mut rng = random::Random::new(708_821 + hops as u32);
        let mut balance = [20usize; 5];
        let mut group: [Vec<Example>; 4] = std::array::from_fn(|_| Vec::with_capacity(100));
        while group[0].len() < 100 {
            let q = Question::random(hops, &mut rng);
            let answer = q.answer()?;
            if balance[answer] == 0 {
                continue;
            }
            let mut reordered = q.clone();
            for table in &mut reordered.tables {
                table.reverse();
            }
            let mut rewired = q.clone();
            let front = &mut rewired.tables[0];
            let mut values: Vec<_> = front.iter().map(|(_, v)| v.clone()).collect();
            values.rotate_left(1);
            for (row, next) in front.iter_mut().zip(values) {
                row.1 = next;
            }
            let mut rotated = q.clone();
            rotated.choices.rotate_left(1);
            if reordered.answer()? != answer
                || rewired.answer()? == answer
                || rotated.answer()? != (answer + 4) % 5
            {
                return Err("paired lookup oracle violated an intervention".into());
            }
            let batch = [q, reordered, rewired, rotated]
                .into_iter()
                .map(|v| encode(&v.render(), v.answer()?, &ids))
                .collect::<Result<Vec<_>>>()?;
            if batch.iter().any(|e| seen.contains(&e.prompt)) {
                continue;
            }
            balance[answer] -= 1;
            for (g, e) in group.iter_mut().zip(batch) {
                seen.insert(e.prompt.clone());
                g.push(e);
            }
        }
        groups.insert(hops, group);
    }
    let bytes = groups
        .iter()
        .map(|(k, g)| {
            (
                k,
                g.each_ref().map(|rows| {
                    rows.iter()
                        .map(|e| (&e.prompt, &e.answer))
                        .collect::<Vec<_>>()
                }),
            )
        })
        .collect::<Vec<_>>();
    let digest = hash(&serde_json::to_vec(&bytes)?);
    Ok(Paired {
        letters: LETTERS.map(|s| ids[s]),
        groups,
        digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rewiring_preserves_the_printed_vocabulary_and_changes_the_answer() {
        let (_, original) = data().unwrap();
        let p = paired(&original).unwrap();
        assert_eq!(
            p.digest,
            "f09d609467a1248ed4cdb48857e283cfbd91ab1c70f9617a2372add7849cf2fe"
        );
        let mut seen: HashSet<_> = original
            .splits
            .values()
            .flatten()
            .map(|row| &row.prompt)
            .collect();
        for group in p.groups.values() {
            assert!(group.iter().all(|g| g.len() == 100));
            for i in 0..100 {
                let mut multiset = group[0][i].prompt.clone();
                multiset.sort_unstable();
                for arm in group {
                    let row = &arm[i];
                    assert!(seen.insert(&row.prompt));
                    let mut actual = row.prompt.clone();
                    actual.sort_unstable();
                    assert_eq!(multiset, actual);
                }
                assert_eq!(group[0][i].answer, group[1][i].answer);
                assert_ne!(group[0][i].answer, group[2][i].answer);
                assert_ne!(group[0][i].answer, group[3][i].answer);
            }
        }
    }
}
