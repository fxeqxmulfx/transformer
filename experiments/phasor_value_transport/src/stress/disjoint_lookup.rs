//! Vary chain length with the same grammar and single-token name strategy from one to forty hops.
use super::*;
#[cfg(test)]
use std::collections::BTreeSet;
use std::collections::HashSet;
pub(super) const NAME: &str = "study-lookup-disjoint-layers";
const WORDS: &str = "<pad> <bos> <eos> Table → Starting at , what do you get ? A . B C D E";
const LETTERS: [&str; 5] = ["A", "B", "C", "D", "E"];
const NAMES: usize = 210;
fn vocabulary() -> Vec<String> {
    WORDS
        .split_whitespace()
        .map(str::to_owned)
        .chain((0..NAMES).map(|i| format!("vem{i}")))
        .collect()
}
struct Set {
    tables: Vec<Vec<(i64, i64)>>,
    starts: [i64; 5],
    options: [i64; 5],
    answers: [usize; 5],
}
fn sample(hops: usize, rng: &mut random::Random) -> Result<Set> {
    if !(1..=40).contains(&hops) {
        return Err("expected 1 to 40 actual tables".into());
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
    fn rows(&self, ids: &BTreeMap<&str, i64>) -> [Example; 5] {
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
            Example::generation(NAME, prompt, vec![ids[LETTERS[self.answers[lane]]]])
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
pub(super) fn dataset() -> Result<(Value, Task)> {
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as i64))
        .collect();
    let mut splits = BTreeMap::new();
    let mut hashes = BTreeMap::new();
    let mut seen = HashSet::new();
    for (name, per_depth, seed) in [("train", 200u32, 3_011_713), ("validation", 40, 3_090_937)] {
        let mut rng = random::Random::new(seed);
        let mut rows = Vec::new();
        for h in 1..=5 {
            for i in 0..per_depth {
                let set = sample(h, &mut rng)?;
                // Uniform row and choice permutations keep the starting key independent of the outer letter quota.
                let lane = set
                    .answers
                    .iter()
                    .position(|&k| k == i as usize % 5)
                    .unwrap();
                let r = set.rows(&ids).into_iter().nth(lane).unwrap();
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
        NAME,
        &words,
        &hashes,
        &case,
        "validation",
    ))?);
    Ok((
        json!({"name":NAME,"tier":"synthetic-study","catalog_sha256":spec,"case":case,"selection":"validation",
        "train_examples":1000,"heldout_examples":200,"split_sha256":hashes}),
        Task {
            vocab: words.len(),
            splits,
            selection: "validation".into(),
        },
    ))
}
fn probes(hops: usize) -> Result<Vec<Example>> {
    if !(1..=40).contains(&hops) {
        return Err("expected one through forty tables".into());
    }
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as i64))
        .collect();
    let mut rng = random::Random::new(3_107_419 + hops as u32);
    let mut rows = Vec::new();
    for _ in 0..20 {
        rows.extend(sample(hops, &mut rng)?.rows(&ids));
    }
    Ok(rows)
}
pub fn score(seed: u32, out: &Path, hops: usize, passes: usize) -> Result<()> {
    let (case, task) = dataset()?;
    let mut cfg = task_config(&case, &task, true)?;
    let (weights, point) = checkpoint(seed, out, &case, &cfg)?;
    let rows = probes(hops)?;
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
    println!(
        "{}",
        json!({"study":NAME,"model":MODEL,"seed":seed,"step":point["step"],"kernite":KERNITE_VERSION,
        "weights_sha256":weight_digest(&weights),"protocol_sha256":case["catalog_sha256"],
        "hops":hops,"shared_reads":passes,"trained_reads":trained,"validation":{"correct":score,"total":val.len()},
        "vocabulary":task.vocab,"name_count":NAMES,"split_sha256":digest(&rows)?,"total":rows.len(),"correct":sets.iter().enumerate().map(|(i,n)|i*n).sum::<usize>(),
        "table_sets":20,"sets_by_correct_answers":sets,"constant_response_sets":constant,"prompt_length":rows[0].prompt.len()})
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
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
        for g in probes(40).unwrap().as_chunks::<5>().0 {
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
