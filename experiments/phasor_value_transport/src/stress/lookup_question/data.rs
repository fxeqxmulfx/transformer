//! A separate word-tokenized synthetic lookup study. Only the complete printed
//! question is encoded for the model. The table oracle supplies the outer target.
use super::super::*;
use std::collections::{BTreeSet, HashSet};
mod paired;
pub(super) mod interior;
pub(super) mod long;
pub(super) mod memory;
pub(super) use paired::paired;

pub(in crate::stress) const NAME: &str = "study-lookup-questions";
const WORDS: [&str; 10] = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
];
pub(super) const LETTERS: [&str; 5] = ["A", "B", "C", "D", "E"];
const NAMES: &str = "Nalo Mira Sori Tavi Yuna vek dax lum pim ros jade quartz amber onyx opal Zevo Cana Feri Leni Korza Pola Veri Zami Nuve Kiri Elsi Juma Heta Xanu Wylo Doru Teso Meli Pras Jovi Kema Orix Tuma Weda Nolvi Zuli Qago Bina Ceti Fomu Ganu Hire Ivas Joli Kupo Lava Mota Niri Oru Pavo Qera Rizu Salo Tora Umi";
const SOURCE: &str = include_str!("../../../../../stress_tests_for_gpt.md");

fn fixture() -> Result<&'static str> {
    SOURCE
        .split_once("### Building a synthetic example")
        .and_then(|(_, s)| s.split_once("```text\n"))
        .and_then(|(_, s)| s.split_once("\n```"))
        .map(|(s, _)| s)
        .ok_or("missing authored lookup question".into())
}

fn lex(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            word.push(c);
        } else {
            if !word.is_empty() {
                out.push(std::mem::take(&mut word));
            }
            if !c.is_whitespace() {
                out.push(c.to_string());
            }
        }
    }
    if !word.is_empty() {
        out.push(word);
    }
    out
}

type Table = Vec<(String, String)>;
#[derive(Clone)]
struct Question {
    tables: Vec<Table>,
    start: String,
    choices: Vec<String>,
}
impl Question {
    fn parse(text: &str) -> Result<Self> {
        let mut blocks: Vec<_> = text.split("\n\n").collect();
        let (query, options) = blocks
            .pop()
            .and_then(|b| b.split_once('\n'))
            .ok_or("missing printed question or letter choices")?;
        if !(1..=WORDS.len()).contains(&blocks.len()) {
            return Err("unsupported number of printed tables".into());
        }
        let start = query
            .strip_prefix("Starting at ")
            .and_then(|s| s.split_once(", "))
            .map(|(key, _)| key.to_owned())
            .ok_or("missing starting key")?;
        let mut tables = Vec::with_capacity(blocks.len());
        for (index, block) in blocks.into_iter().enumerate() {
            let (title, rows) = block.split_once('\n').ok_or("missing printed table rows")?;
            if title != format!("Table {}", index + 1) {
                return Err("unexpected printed table index".into());
            }
            let mut rows = rows.split_whitespace();
            let mut table = Vec::new();
            while let Some(key) = rows.next() {
                if rows.next() != Some("→") {
                    return Err("missing mapping arrow".into());
                }
                let value = rows.next().ok_or("missing printed value")?;
                table.push((key.to_owned(), value.to_owned()));
            }
            tables.push(table);
        }
        let mut options = options.split_whitespace();
        let mut choices = Vec::with_capacity(LETTERS.len());
        for letter in LETTERS {
            if options.next().and_then(|s| s.strip_suffix('.')) != Some(letter) {
                return Err("missing or reordered choice letter".into());
            }
            choices.push(options.next().ok_or("missing printed choice")?.to_owned());
        }
        let question = Self {
            tables,
            start,
            choices,
        };
        if question.render() != text {
            return Err("printed question differs from its declared table count or choices".into());
        }
        Ok(question)
    }

    fn random(hops: usize, rng: &mut random::Random) -> Self {
        let mut names: Vec<_> = NAMES.split_whitespace().collect();
        rng.shuffle(&mut names);
        let pools = &names[..5 * (hops + 1)];
        let start = pools[rng.below(5) as usize].to_string();
        let mut tables = Vec::with_capacity(hops);
        for k in 0..hops {
            let left = &pools[k * 5..][..5];
            let mut right = pools[(k + 1) * 5..][..5].to_vec();
            rng.shuffle(&mut right);
            let mut table: Table = left
                .iter()
                .zip(right)
                .map(|(&a, b)| (a.to_string(), b.to_string()))
                .collect();
            rng.shuffle(&mut table);
            tables.push(table);
        }
        let mut choices: Vec<_> = pools[hops * 5..].iter().map(|v| v.to_string()).collect();
        rng.shuffle(&mut choices);
        Self {
            tables,
            start,
            choices,
        }
    }
    fn answer(&self) -> Result<usize> {
        let mut value = self.start.as_str();
        for table in &self.tables {
            let mut matches = table.iter().filter(|(k, _)| k == value);
            let next = matches.next().ok_or("missing question key")?.1.as_str();
            if matches.next().is_some() {
                return Err("ambiguous table key".into());
            }
            value = next;
        }
        let mut matches = self
            .choices
            .iter()
            .enumerate()
            .filter(|(_, v)| v.as_str() == value);
        let answer = matches.next().ok_or("missing answer choice")?.0;
        if matches.next().is_some() {
            return Err("duplicate answer choice".into());
        }
        Ok(answer)
    }
    fn render(&self) -> String {
        let mut blocks = Vec::new();
        for (i, t) in self.tables.iter().enumerate() {
            let rows = t
                .iter()
                .map(|(k, v)| format!("{k} → {v}"))
                .collect::<Vec<_>>()
                .join("   ");
            blocks.push(format!("Table {}\n{rows}", i + 1));
        }
        let n = self.tables.len();
        let word = WORDS[n - 1];
        let noun = if n == 1 { "lookup" } else { "lookups" };
        let options = LETTERS
            .iter()
            .zip(&self.choices)
            .map(|(l, v)| format!("{l}. {v}"))
            .collect::<Vec<_>>()
            .join("   ");
        blocks.push(format!(
            "Starting at {}, what do you get after {word} {noun}?\n{options}",
            self.start
        ));
        blocks.join("\n\n")
    }
}

pub(super) fn vocabulary() -> Vec<String> {
    let mut words: BTreeSet<_> =
        lex("Table Starting at , what do you get after lookup lookups ? A . B C D E →")
            .into_iter()
            .collect();
    words.extend(NAMES.split_whitespace().chain(WORDS).map(str::to_owned));
    words.extend((1..=10).map(|i| i.to_string()));
    ["<pad>", "<bos>", "<eos>"]
        .into_iter()
        .map(str::to_owned)
        .chain(words)
        .collect()
}

fn encode(text: &str, answer: usize, ids: &BTreeMap<&str, i64>) -> Result<Example> {
    let mut prompt = vec![1];
    for token in lex(text) {
        prompt.push(*ids.get(token.as_str()).ok_or("unknown question token")?);
    }
    Ok(Example::generation(
        NAME,
        prompt,
        vec![ids[LETTERS[answer]]],
    ))
}

fn balanced(
    hops: usize,
    per_letter: usize,
    seed: u32,
    ids: &BTreeMap<&str, i64>,
    seen: &mut HashSet<Vec<i64>>,
) -> Result<Vec<Example>> {
    let mut rng = random::Random::new(seed);
    let mut left = [per_letter; 5];
    let mut rows = Vec::with_capacity(5 * per_letter);
    while rows.len() < 5 * per_letter {
        let q = Question::random(hops, &mut rng);
        let label = q.answer()?;
        if left[label] == 0 {
            continue;
        }
        let row = encode(&q.render(), label, ids)?;
        if seen.insert(row.prompt.clone()) {
            left[label] -= 1;
            rows.push(row);
        }
    }
    rng.shuffle(&mut rows);
    Ok(rows)
}

pub(super) fn split_digest(rows: &[Example]) -> Result<String> {
    let payload: Vec<_> = rows.iter().map(|r| (&r.prompt, &r.answer)).collect();
    Ok(hash(&serde_json::to_vec(&payload)?))
}

/// Complete the test grid after authenticating the training checkpoint.
/// Each depth uses its own generator seed and balances all five letters.
pub(super) fn complete_hops(task: &mut Task) -> Result<BTreeMap<usize, String>> {
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as i64))
        .collect();
    let mut seen = task
        .splits
        .values()
        .flatten()
        .map(|e| e.prompt.clone())
        .collect();
    let mut digests = BTreeMap::new();
    for hops in 1..=WORDS.len() {
        let entry = task.splits.entry(format!("test/hops-{hops}"));
        let rows = match entry {
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(balanced(hops, 20, 378_609 + hops as u32, &ids, &mut seen)?)
            }
        };
        digests.insert(hops, split_digest(rows)?);
    }
    Ok(digests)
}

pub(in crate::stress) fn data() -> Result<(Value, Task)> {
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as i64))
        .collect();
    let printed = fixture()?;
    let mira = encode(printed, Question::parse(printed)?.answer()?, &ids)?;
    let mut seen = HashSet::from([mira.prompt.clone()]);
    let mut splits = BTreeMap::from([("test/mira".to_string(), vec![mira])]);
    for (split, per, seed) in [("train", 100, 104_071u32), ("validation", 20, 208_133)] {
        let mut one = balanced(1, per, seed, &ids, &mut seen)?;
        one.extend(balanced(2, per, seed + 1, &ids, &mut seen)?);
        random::Random::new(seed + 2).shuffle(&mut one);
        splits.insert(split.to_string(), one);
    }
    for hops in [1, 2, 3, 5, 10] {
        let name = format!("test/hops-{hops}");
        splits.insert(
            name,
            balanced(hops, 20, 378_609 + hops as u32, &ids, &mut seen)?,
        );
    }
    let mut digests = BTreeMap::new();
    for (name, rows) in &splits {
        digests.insert(name, split_digest(rows)?);
    }
    let case = json!({"width":64,"heads":2,"layers":2,"batch":16,"max_updates":3000,"observe":50});
    let fingerprint = hash(&serde_json::to_vec(&(
        NAME,
        &words,
        &digests,
        &case,
        "validation",
    ))?);
    let row = json!({"name":NAME,"tier":"synthetic-study","catalog_sha256":fingerprint,"selection":"validation",
        "train_examples":splits["train"].len(),"heldout_examples":splits["validation"].len(),"split_sha256":digests,
        "case":case});
    Ok((
        row,
        Task {
            splits,
            vocab: words.len(),
            selection: "validation".into(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_grid_preserves_the_frozen_splits_and_adds_five_balanced_depths() {
        let (base, mut task) = data().unwrap();
        let hashes = complete_hops(&mut task).unwrap();
        assert_eq!(hashes.len(), 10);
        for (split, digest) in base["split_sha256"].as_object().unwrap() {
            assert_eq!(
                split_digest(&task.splits[split]).unwrap(),
                digest.as_str().unwrap()
            );
        }
        let words = vocabulary();
        let mut seen = HashSet::new();
        for (name, rows) in &task.splits {
            for r in rows {
                assert!(seen.insert(&r.prompt));
            }
            if name.starts_with("test/hops-") {
                let mut counts = [0; 5];
                for e in rows {
                    let label = &words[e.answer[0] as usize];
                    counts[LETTERS.iter().position(|x| x == label).unwrap()] += 1;
                }
                assert_eq!(counts, [20; 5]);
            }
        }
        assert_eq!(complete_hops(&mut task).unwrap(), hashes);
    }
    #[test]
    fn every_split_has_fresh_unannotated_questions_and_balanced_letters() {
        let (row, t) = data().unwrap();
        let cfg = task_config(&row, &t, true).unwrap();
        assert_eq!(cfg.warmup, schedule::Schedule::new(3000).unwrap().warmup);
        assert_eq!(t.splits["train"].len(), 1000);
        assert_eq!(t.splits["validation"].len(), 200);
        let words = vocabulary();
        let mut seen = HashSet::new();
        for (split, rows) in &t.splits {
            let mut counts = [0; 5];
            for e in rows {
                assert_eq!(e.tokens, e.prompt);
                assert!(e.targets[..e.targets.len() - 1].iter().all(|&x| x == -100));
                assert_eq!(e.targets.last(), e.answer.last());
                assert_eq!(e.limit, 1);
                assert!(seen.insert(&e.prompt));
                counts[LETTERS
                    .iter()
                    .position(|&s| s == words[e.answer[0] as usize])
                    .unwrap()] += 1;
            }
            if split != "test/mira" {
                assert!(counts.iter().all(|&n| n * 5 == rows.len()));
            }
        }
        let mut q = Question::parse(fixture().unwrap()).unwrap();
        assert_eq!(q.answer().unwrap(), 2);
        q.tables.iter_mut().for_each(|table| table.reverse());
        q.choices.rotate_left(1);
        assert_eq!(Question::parse(&q.render()).unwrap().answer().unwrap(), 1);
        let mut rng = random::Random::new(3301);
        for hops in 1..=WORDS.len() {
            let q = Question::random(hops, &mut rng);
            let decoded = Question::parse(&q.render()).unwrap();
            assert_eq!(decoded.answer().unwrap(), q.answer().unwrap());
        }
        assert_eq!(
            row["catalog_sha256"],
            "ba0868a867169e9ef4601eb37076159e779fc72c70abf26c5fa44b5aea307f31"
        );
    }
}
