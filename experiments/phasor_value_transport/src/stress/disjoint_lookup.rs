//! Vary chain length up to the capacity of one fixed, single-token name alphabet.
use super::*;
pub mod budgets;
#[cfg(test)]
use std::collections::BTreeSet;
use std::collections::HashSet;
pub(super) const NAME: &str = "study-lookup-disjoint-layers";

#[derive(Clone, Copy)]
pub(super) struct Training {
    name: &'static str,
    depths: &'static [usize],
}
impl Training {
    pub(super) fn named(name: &str) -> Option<Self> {
        match name {
            NAME => Some(Self {
                name: NAME,
                depths: &[1, 2, 3, 4, 5],
            }),
            "study-lookup-disjoint-through-10" => Some(Self {
                name: "study-lookup-disjoint-through-10",
                depths: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }),
            "study-lookup-disjoint-through-20" => Some(Self {
                name: "study-lookup-disjoint-through-20",
                depths: &[1, 2, 3, 4, 5, 8, 10, 12, 16, 20],
            }),
            _ => None,
        }
    }
}
const WORDS: &str = "<pad> <bos> <eos> Table → Starting at , what do you get ? A . B C D E";
const LETTERS: [&str; 5] = ["A", "B", "C", "D", "E"];
const NAMES: usize = 210;
const MAX_HOPS: usize = NAMES / LETTERS.len() - 1;
fn vocabulary() -> Vec<String> {
    WORDS
        .split_whitespace()
        .map(str::to_owned)
        .chain((0..NAMES).map(|i| format!("vem{i}")))
        .collect()
}
#[derive(Clone)]
struct Set {
    tables: Vec<Vec<(i64, i64)>>,
    starts: [i64; 5],
    options: [i64; 5],
    answers: [usize; 5],
}
fn sample(hops: usize, rng: &mut random::Random) -> Result<Set> {
    if !(1..=MAX_HOPS).contains(&hops) {
        return Err(format!("expected one through {MAX_HOPS} disjoint tables").into());
    }
    let first = WORDS.split_whitespace().count() as i64;
    let mut names: Vec<_> = (first..first + NAMES as i64).collect();
    rng.shuffle(&mut names);
    let starts: [i64; 5] = names[..5].try_into()?;
    let mut layer = starts;
    let mut tables = Vec::new();
    for step in 1..=hops {
        let next: [i64; 5] = names[5 * step..5 * step + 5].try_into()?;
        let mut table: Vec<_> = layer.into_iter().zip(next).collect();
        rng.shuffle(&mut table);
        tables.push(table);
        layer = next;
    }
    let mut options = layer;
    rng.shuffle(&mut options);
    let answers = std::array::from_fn(|i| options.iter().position(|&x| x == layer[i]).unwrap());
    Ok(Set {
        tables,
        starts,
        options,
        answers,
    })
}
impl Set {
    fn rows(&self, ids: &BTreeMap<&str, i64>, task: &str) -> [Example; 5] {
        let mut prefix = vec![ids["<bos>"]];
        for t in &self.tables {
            prefix.push(ids["Table"]);
            for &(a, b) in t {
                prefix.extend([a, ids["→"], b]);
            }
        }
        std::array::from_fn(|lane| {
            let mut prompt = prefix.clone();
            prompt.extend([
                ids["Starting"],
                ids["at"],
                self.starts[lane],
                ids[","],
                ids["what"],
                ids["do"],
                ids["you"],
                ids["get"],
                ids["?"],
            ]);
            for (letter, key) in LETTERS.into_iter().zip(self.options) {
                prompt.extend([ids[letter], ids["."], key]);
            }
            Example::generation(task, prompt, vec![ids[LETTERS[self.answers[lane]]]])
        })
    }
}
fn digest(rows: &[Example]) -> Result<String> {
    Ok(hash(&serde_json::to_vec(
        &rows
            .iter()
            .map(|r| (&r.prompt, &r.answer))
            .collect::<Vec<_>>(),
    )?))
}
pub(super) fn dataset(training: Training) -> Result<(Value, Task)> {
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as i64))
        .collect();
    let mut splits = BTreeMap::new();
    let mut hashes = BTreeMap::new();
    let mut seen = HashSet::new();
    for (name, count, seed) in [("train", 1000, 3_011_713), ("validation", 200, 3_090_937)] {
        let mut rng = random::Random::new(seed);
        let mut rows = Vec::new();
        for &h in training.depths {
            for i in 0..count / training.depths.len() {
                let set = sample(h, &mut rng)?;
                // Uniform row and choice permutations keep the starting key independent of the outer letter quota.
                let lane = set.answers.iter().position(|&k| k == i % 5).unwrap();
                let r = set.rows(&ids, training.name).into_iter().nth(lane).unwrap();
                if !seen.insert(r.prompt.clone()) {
                    return Err("duplicate synthetic prompt".into());
                }
                rows.push(r);
            }
        }
        rng.shuffle(&mut rows);
        hashes.insert(name, digest(&rows)?);
        splits.insert(name.into(), rows);
    }
    let case = json!({"width":64,"heads":2,"layers":2,"batch":16,"max_updates":3000,"observe":50});
    let spec = hash(&serde_json::to_vec(&(
        training.name,
        &words,
        &hashes,
        &case,
        "validation",
    ))?);
    Ok((
        json!({"name":training.name,"tier":"synthetic-study","catalog_sha256":spec,"case":case,"selection":"validation",
        "train_examples":1000,"heldout_examples":200,"split_sha256":hashes}),
        Task {
            vocab: words.len(),
            splits,
            selection: "validation".into(),
        },
    ))
}
pub(super) fn probes(hops: usize, task: &str) -> Result<Vec<Example>> {
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as i64))
        .collect();
    let mut rng = random::Random::new(3_107_419u32.wrapping_add(hops as u32));
    let mut rows = Vec::new();
    for _ in 0..20 {
        rows.extend(sample(hops, &mut rng)?.rows(&ids, task));
    }
    Ok(rows)
}
pub fn score(seed: u32, out: &Path, hops: usize, passes: usize, name: Option<&str>) -> Result<()> {
    let training =
        Training::named(name.unwrap_or(NAME)).ok_or("unknown disjoint-layer training study")?;
    let (case, task) = dataset(training)?;
    let mut cfg = task_config(&case, &task, true)?;
    let (weights, point) = checkpoint(seed, out, &case, &cfg)?;
    let rows = probes(hops, training.name)?;
    cfg.max_batch = task_config(&case, &task, false)?.max_batch;
    cfg.max_length = rows[0].prompt.len().max(cfg.max_length);
    let trained = cfg.head.peer_read().and_then(|r| r.path).map(|r| r.passes);
    lookup_question::hop_sweep::set_budget(&mut cfg, passes)?;
    let mut engine = Engine::for_scoring(cfg, &weights)?;
    let val = &task.splits["validation"];
    let score = lookup_question::hop_sweep::generated(&mut engine, val)?
        .iter()
        .zip(val)
        .filter(|(p, e)| **p == Some(e.answer[0]))
        .count();
    if trained == Some(passes) && point["correct"] != score {
        return Err("scheduled validation replay changed".into());
    }
    let predictions = lookup_question::hop_sweep::generated(&mut engine, &rows)?;
    let mut sets = [0; 6];
    let mut constant = 0;
    let (questions, tail) = rows.as_chunks::<5>();
    let (answers, extra) = predictions.as_chunks::<5>();
    if !tail.is_empty() || !extra.is_empty() || questions.len() != answers.len() {
        return Err("incomplete group of five starts".into());
    }
    for (g, r) in questions.iter().zip(answers) {
        let right = g
            .iter()
            .zip(r)
            .filter(|(e, p)| **p == Some(e.answer[0]))
            .count();
        sets[right] += 1;
        constant += usize::from(r.iter().all(|p| *p == r[0]));
    }
    let mut report = json!({"study":training.name,"model":MODEL,"seed":seed,"step":point["step"],"kernite":KERNITE_VERSION,
        "weights_sha256":weight_digest(&weights),"protocol_sha256":case["catalog_sha256"],
        "hops":hops,"shared_reads":passes,"trained_reads":trained,"validation":{"correct":score,"total":val.len()},
        "vocabulary":task.vocab,"name_count":NAMES,"split_sha256":digest(&rows)?,"total":rows.len(),"correct":sets.iter().enumerate().map(|(i,n)|i*n).sum::<usize>(),
        "table_sets":20,"sets_by_correct_answers":sets,"constant_response_sets":constant,"prompt_length":rows[0].prompt.len()});
    if training.name != NAME {
        report["training_hops"] = json!(training.depths);
    }
    println!("{report}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn curricula_balance_hops_and_letters_and_reuse_independent_table_sets() {
        for name in [
            NAME,
            "study-lookup-disjoint-through-10",
            "study-lookup-disjoint-through-20",
        ] {
            let training = Training::named(name).unwrap();
            let (manifest, task) = dataset(training).unwrap();
            if name == NAME {
                assert_eq!(
                    manifest["catalog_sha256"],
                    "0bd7bf4cf889a8b6bdecab3e02a9dcaa19f43a4d7a3f2c008f2bcf23accd287b"
                );
            }
            let mut seen = HashSet::new();
            for (split, total) in [("train", 1000), ("validation", 200)] {
                let mut cells = BTreeMap::new();
                assert_eq!(task.splits[split].len(), total);
                for row in &task.splits[split] {
                    let hops = (row.prompt.len() - 25) / 16;
                    assert!(training.depths.contains(&hops));
                    assert_eq!(row.task, name);
                    assert_eq!(row.targets.iter().filter(|&&v| v != -100).count(), 1);
                    assert!(seen.insert(row.prompt.clone()));
                    *cells.entry((hops, row.answer[0])).or_insert(0usize) += 1;
                }
                assert_eq!(cells.len(), training.depths.len() * 5);
                assert!(
                    cells
                        .values()
                        .all(|&n| n == total / (5 * training.depths.len()))
                );
            }
            let test = probes(10, name).unwrap();
            assert_eq!(
                digest(&test).unwrap(),
                "e268a2ab58511be4644952327ecc7db74b20107b5f410f313f9a7410564aa4fb"
            );
            assert!(test.iter().all(|r| !seen.contains(&r.prompt)));
            assert!(
                probes(40, name)
                    .unwrap()
                    .iter()
                    .all(|r| !seen.contains(&r.prompt))
            );
        }
    }
    #[test]
    fn the_last_supported_length_uses_the_entire_frozen_alphabet() {
        let mut rng = random::Random::new(58);
        let set = sample(41, &mut rng).unwrap();
        let names: BTreeSet<_> = set.starts.into_iter()
            .chain(set.tables.iter().flatten().map(|&(_, dest)| dest)).collect();
        assert_eq!(names.len(), NAMES);
        let questions = probes(41, NAME).unwrap();
        assert_eq!(questions.len(), 100);
        assert!(questions.iter().all(|r| r.prompt.len() == 681));
        assert!(sample(42, &mut rng).is_err());
        assert!(probes(0, NAME).is_err());
    }

    #[test]
    fn forty_hops_use_205_different_names_and_only_the_final_letter_as_target() {
        let t = sample(40, &mut random::Random::new(58)).unwrap();
        let mut names: BTreeSet<_> = t.starts.into_iter().collect();
        for tab in &t.tables {
            for &(_, v) in tab {
                assert!(names.insert(v));
            }
        }
        assert_eq!(names.len(), 205);
        for lane in 0..5 {
            let mut key = t.starts[lane];
            for tab in &t.tables {
                key = tab.iter().find(|(a, _)| *a == key).unwrap().1;
            }
            assert_eq!(key, t.options[t.answers[lane]]);
        }
        for g in probes(40, NAME).unwrap().as_chunks::<5>().0 {
            assert_eq!(
                g.iter().map(|r| r.answer[0]).collect::<BTreeSet<_>>().len(),
                5
            );
            for r in g {
                assert_eq!(r.prompt.len(), 665);
                assert_eq!(r.tokens, r.prompt);
                assert_eq!(r.targets.iter().filter(|&&x| x != -100).count(), 1);
            }
        }
    }
}
