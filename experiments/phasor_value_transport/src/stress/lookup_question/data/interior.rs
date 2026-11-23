//! Token-balanced counterfactuals on the reached edge of each individual table.
use super::*;

pub(in crate::stress::lookup_question) struct Group {
    pub arms: BTreeMap<String, Vec<Example>>,
    pub digests: BTreeMap<String, String>,
}

impl Question {
    fn reached_key(&self, depth: usize) -> Result<&str> {
        let mut key = self.start.as_str();
        for table in &self.tables[..depth] {
            key = &table
                .iter()
                .find(|(k, _)| k == key)
                .ok_or("missing interior key")?
                .1;
        }
        Ok(key)
    }

    // Swap one on-path output with one off-path output. Exactly two printed cells
    // change, and the off-path input is unreachable before this intervention.
    fn change_edge(&self, depth: usize, target: usize) -> Result<Self> {
        let key = self.reached_key(depth)?;
        let table = &self.tables[depth];
        let row = table
            .iter()
            .position(|(k, _)| k == key)
            .ok_or("missing reached row")?;
        for other in 0..table.len() {
            if other == row {
                continue;
            }
            let mut candidate = self.clone();
            candidate.tables[depth][row].1.clone_from(&table[other].1);
            candidate.tables[depth][other].1.clone_from(&table[row].1);
            if candidate.answer()? == target {
                return Ok(candidate);
            }
        }
        Err("no single relevant rewire gives the requested alternative".into())
    }
}

pub(in crate::stress::lookup_question) fn group(task: &Task, hops: usize) -> Result<Group> {
    if !(1..=WORDS.len()).contains(&hops) {
        return Err("choose a depth from one through ten".into());
    }
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as i64))
        .collect();
    let mut seen: HashSet<_> = task
        .splits
        .values()
        .flatten()
        .map(|e| e.prompt.clone())
        .collect();
    let mut questions = random::Random::new(840_013 + hops as u32);
    let mut row_order = random::Random::new(847_091 + hops as u32);
    let mut choice_order = random::Random::new(851_137 + hops as u32);
    let mut counts = [0; 5];
    let mut arms = BTreeMap::<String, Vec<Example>>::new();
    while counts.iter().sum::<usize>() < 100 {
        let q = Question::random(hops, &mut questions);
        let gold = q.answer()?;
        if counts[gold] == 20 {
            continue;
        }
        let mut rows = q.clone();
        for t in &mut rows.tables {
            row_order.shuffle(t);
        }
        let mut options = q.clone();
        choice_order.shuffle(&mut options.choices);
        let mut family = vec![
            ("original".to_owned(), q.clone()),
            ("rows_shuffled".into(), rows),
            ("choices_shuffled".into(), options),
        ];
        for depth in 0..hops {
            let target = (gold + 1 + (counts[gold] + depth) % 4) % 5;
            family.push((
                format!("rewire-{}", depth + 1),
                q.change_edge(depth, target)?,
            ));
        }
        let mut fresh = HashSet::new();
        let mut encoded = Vec::with_capacity(family.len());
        for (name, item) in family {
            let e = encode(&item.render(), item.answer()?, &ids)?;
            if seen.contains(&e.prompt) || !fresh.insert(e.prompt.clone()) {
                break;
            }
            encoded.push((name, e));
        }
        if encoded.len() != hops + 3 {
            continue;
        }
        counts[gold] += 1;
        seen.extend(fresh);
        for (name, e) in encoded {
            arms.entry(name).or_default().push(e);
        }
    }
    let digests = arms
        .iter()
        .map(|(k, rows)| Ok((k.clone(), split_digest(rows)?)))
        .collect::<Result<_>>()?;
    Ok(Group { arms, digests })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn each_rewrite_changes_only_the_reached_edge_and_one_balancing_cell() {
        let mut rng = random::Random::new(3289);
        for hops in 1..=10 {
            let original = Question::random(hops, &mut rng);
            for depth in 0..hops {
                let changed = original
                    .change_edge(depth, (original.answer().unwrap() + 1) % 5)
                    .unwrap();
                assert_eq!(
                    original.reached_key(depth).unwrap(),
                    changed.reached_key(depth).unwrap()
                );
                assert_eq!(original.tables[..depth], changed.tables[..depth]);
                assert_eq!(original.tables[depth + 1..], changed.tables[depth + 1..]);
                assert_eq!(original.choices, changed.choices);
                let cells: Vec<_> = original.tables[depth]
                    .iter()
                    .zip(&changed.tables[depth])
                    .filter(|(a, b)| a != b)
                    .collect();
                assert_eq!(cells.len(), 2);
                assert!(
                    cells
                        .iter()
                        .any(|(a, _)| a.0 == original.reached_key(depth).unwrap())
                );
                assert_eq!(cells[0].0.1, cells[1].1.1);
                assert_eq!(cells[1].0.1, cells[0].1.1);
            }
        }
    }
    #[test]
    fn independent_controls_and_each_changed_answer_are_balanced_and_disjoint() {
        let (_, mut t) = data().unwrap();
        complete_hops(&mut t).unwrap();
        let g = group(&t, 5).unwrap();
        let mut seen: HashSet<_> = t.splits.values().flatten().map(|e| &e.prompt).collect();
        for (arm, rows) in &g.arms {
            assert_eq!(rows.len(), 100);
            let mut histogram = BTreeMap::new();
            for (base, e) in g.arms["original"].iter().zip(rows) {
                assert!(seen.insert(&e.prompt));
                let mut a = e.prompt.clone();
                let mut b = base.prompt.clone();
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(a, b);
                assert_eq!(e.targets.iter().filter(|&&y| y != -100).count(), 1);
                if arm.starts_with("rewire-") {
                    assert_ne!(base.answer, e.answer);
                }
                if arm == "rows_shuffled" {
                    assert_eq!(base.answer, e.answer);
                }
                *histogram.entry(&e.answer).or_insert(0_usize) += 1;
            }
            if arm != "choices_shuffled" {
                assert!(histogram.len() == 5 && histogram.values().all(|&n| n == 20));
            }
        }
    }
}
