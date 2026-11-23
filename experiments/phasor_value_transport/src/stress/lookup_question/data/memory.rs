//! Matched changes to a requested table and to a later distractor table.
use super::*;

pub(in crate::stress::lookup_question) struct Family {
    pub arms: BTreeMap<&'static str, Vec<Example>>,
    pub last_key_letters: Vec<i64>,
    pub digests: BTreeMap<&'static str, String>,
}

fn with_tail(q: &Question, tail: &[(String, String)]) -> Result<String> {
    let text = q.render();
    let (tables, query) = text.rsplit_once("\n\n").ok_or("missing printed question")?;
    let rows = tail
        .iter()
        .map(|(k, v)| format!("{k} → {v}"))
        .collect::<Vec<_>>()
        .join("   ");
    Ok(format!("{tables}\n\nTable A\n{rows}\n\n{query}"))
}

// The overwrite arms allow the same key to occur twice in a numbered table.
// Their oracle reads the latest binding within each requested table.
fn latest_answer(q: &Question) -> Result<usize> {
    let mut key = q.start.as_str();
    for table in &q.tables {
        key = &table
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .ok_or("missing key in overwrite control")?
            .1;
    }
    q.choices
        .iter()
        .position(|value| value == key)
        .ok_or("missing overwrite choice".into())
}

pub(in crate::stress::lookup_question) fn family(
    task: &Task,
    hops: usize,
    edge: usize,
) -> Result<Family> {
    if !(1..=WORDS.len()).contains(&hops) || !(1..=hops).contains(&edge) {
        return Err("choose an existing hop and edge".into());
    }
    let depth = edge - 1;
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as i64))
        .collect();
    let mut rng = random::Random::new(966_157 + 31 * hops as u32 + edge as u32);
    let mut seen: HashSet<_> = task
        .splits
        .values()
        .flatten()
        .map(|e| e.prompt.clone())
        .collect();
    let mut arms: BTreeMap<_, Vec<_>> = BTreeMap::new();
    let mut last_key_letters = Vec::new();
    let mut count = [0; 5];
    while last_key_letters.len() < 100 {
        let mut q = Question::random(hops, &mut rng);
        let gold = q.answer()?;
        if count[gold] == 20 {
            continue;
        }
        let alternate = (gold + 1 + count[gold] % 4) % 5;
        let mut possible = (0..5)
            .filter(|&i| i != gold && i != alternate)
            .collect::<Vec<_>>();
        rng.shuffle(&mut possible);
        let lure = possible[0];
        let reached = q.reached_key(depth)?.to_owned();
        let at = q.tables[depth]
            .iter()
            .position(|(k, _)| k == &reached)
            .ok_or("missing on-path row")?;
        let absent = NAMES
            .split_whitespace()
            .find(|&name| {
                q.tables
                    .iter()
                    .flatten()
                    .all(|(k, v)| k != name && v != name)
            })
            .ok_or("no unused replacement key")?;
        let mut changed = q.change_edge(depth, alternate)?;
        q.choices[lure].clone_from(&reached);
        changed.choices[lure].clone_from(&reached);
        let old = &q.tables[depth];
        let new = &changed.tables[depth];
        let mut old_then_new = q.clone();
        old_then_new.tables[depth].push(new[at].clone());
        let mut new_then_old = changed.clone();
        new_then_old.tables[depth].push(old[at].clone());
        if latest_answer(&old_then_new)? != alternate || latest_answer(&new_then_old)? != gold {
            return Err("the last in-table binding must determine the answer".into());
        }
        let mut missing = q.clone();
        missing.tables[depth][at].0 = absent.into();
        let mut off = q.clone();
        let nrows = off.tables[depth].len();
        off.tables[depth][(at + 1) % nrows].0 = absent.into();
        if missing.answer().is_ok() || off.answer()? != gold || q.answer()? != gold {
            return Err("missing-edge control changed the wrong route".into());
        }
        // For the missing link only, the target is a diagnostic letter naming
        // the last reachable key. The complete-chain oracle has no answer.
        let cases = [
            ("original", q.render(), gold),
            ("active_rewired", changed.render(), alternate),
            ("old_active_old_tail", with_tail(&q, old)?, gold),
            ("old_active_new_tail", with_tail(&q, new)?, gold),
            ("old_overwritten_inside", old_then_new.render(), alternate),
            (
                "old_with_same_row_outside",
                with_tail(&q, &new[at..=at])?,
                gold,
            ),
            ("new_overwritten_inside", new_then_old.render(), gold),
            (
                "new_with_same_row_outside",
                with_tail(&changed, &old[at..=at])?,
                alternate,
            ),
            ("new_active_old_tail", with_tail(&changed, old)?, alternate),
            ("new_active_new_tail", with_tail(&changed, new)?, alternate),
            ("missing_reached_edge", missing.render(), lure),
            ("missing_off_path_edge", off.render(), gold),
        ];
        let mut fresh = HashSet::new();
        let mut batch = Vec::new();
        let expected = cases.len();
        for (name, text, label) in cases {
            let e = encode(&text, label, &ids)?;
            if seen.contains(&e.prompt) || !fresh.insert(e.prompt.clone()) {
                break;
            }
            batch.push((name, e));
        }
        if batch.len() != expected {
            continue;
        }
        count[gold] += 1;
        seen.extend(fresh);
        last_key_letters.push(ids[LETTERS[lure]]);
        for (name, e) in batch {
            arms.entry(name).or_default().push(e);
        }
    }
    let digests = arms
        .iter()
        .map(|(&k, g)| Ok((k, split_digest(g)?)))
        .collect::<Result<_>>()?;
    Ok(Family {
        arms,
        last_key_letters,
        digests,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conflicting_tail_is_outside_the_query_and_missing_links_have_a_displayed_lure() {
        let (_, mut task) = data().unwrap();
        complete_hops(&mut task).unwrap();
        let f = family(&task, 3, 2).unwrap();
        for i in 0..100 {
            let row = |key: &str| &f.arms[key][i];
            assert_eq!(
                row("old_overwritten_inside").answer,
                row("active_rewired").answer
            );
            assert_eq!(
                row("old_with_same_row_outside").answer,
                row("original").answer
            );
            assert_eq!(row("new_overwritten_inside").answer, row("original").answer);
            assert_eq!(
                row("new_with_same_row_outside").answer,
                row("active_rewired").answer
            );
            assert_eq!(
                row("old_active_old_tail").answer,
                row("old_active_new_tail").answer
            );
            assert_eq!(
                row("new_active_old_tail").answer,
                row("new_active_new_tail").answer
            );
            assert_ne!(row("active_rewired").answer, row("original").answer);
            assert_eq!(row("original").answer, row("missing_off_path_edge").answer);
            assert_ne!(row("original").answer, row("missing_reached_edge").answer);
            assert_ne!(
                row("active_rewired").answer,
                row("missing_reached_edge").answer
            );
            assert_eq!(row("missing_reached_edge").answer[0], f.last_key_letters[i]);
            assert_eq!(
                row("old_active_old_tail")
                    .prompt
                    .iter()
                    .zip(&row("old_active_new_tail").prompt)
                    .filter(|(a, b)| a != b)
                    .count(),
                2
            );
            assert_eq!(
                row("old_active_new_tail")
                    .prompt
                    .iter()
                    .zip(&row("new_active_new_tail").prompt)
                    .filter(|(a, b)| a != b)
                    .count(),
                2
            );
        }
    }
}
