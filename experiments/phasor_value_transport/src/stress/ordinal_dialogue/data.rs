//! Training and evaluation use text alone. Metadata stays in the outer generator.
use super::super::*;
use std::collections::{BTreeSet, HashSet};
pub(in crate::stress) const NAME: &str = "study-ordinal-dialogue";
const ORDINALS: [&str; 8] = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
];
const TOPICS: [&str; 6] = ["moss", "rivers", "bees", "clouds", "bells", "trains"];
const FORMS: [&str; 4] = ["poem", "story", "riddle", "song"];
const VALUES: &str = "amber blue coral dune elm fern garnet hazel iris jade kelp linen maple nutmeg olive pearl quartz ruby sage teal umber violet wheat zinc aqua birch cedar flax ginger indigo lilac ochre";
fn vocabulary() -> Vec<&'static str> {
    let fixed = "User : write a about . Assistant the word is for , give me choose I repeat which was requested";
    let symbols: BTreeSet<_> = fixed
        .split_whitespace()
        .chain(ORDINALS)
        .chain(TOPICS)
        .chain(FORMS)
        .chain(VALUES.split_whitespace())
        .collect();
    ["<pad>", "<bos>", "<eos>"]
        .into_iter()
        .chain(symbols)
        .collect()
}
fn text(ids: &BTreeMap<&str, i64>, printed: &str) -> Result<Vec<i64>> {
    printed
        .split_whitespace()
        .map(|s| {
            ids.get(s)
                .copied()
                .ok_or_else(|| format!("unknown dialogue token {s}").into())
        })
        .collect()
}
#[derive(Clone, Copy)]
struct Turn<'a> {
    topic: &'a str,
    form: &'a str,
    reply: &'a str,
}
fn examples(
    ids: &BTreeMap<&str, i64>,
    rng: &mut random::Random,
    distractors: usize,
    seen: &mut HashSet<Vec<i64>>,
) -> Result<Vec<Example>> {
    let mut topics = TOPICS;
    let mut forms = FORMS;
    let mut values: Vec<_> = VALUES.split_whitespace().collect();
    rng.shuffle(&mut topics);
    rng.shuffle(&mut forms);
    rng.shuffle(&mut values);
    let (topic, form) = (topics[0], forms[0]);
    let mut turns: Vec<_> = values[..8]
        .iter()
        .map(|&reply| Turn { topic, form, reply })
        .collect();
    for (i, &reply) in values[8..8 + distractors].iter().enumerate() {
        let (topic, form) = if i % 2 == 0 {
            (topic, forms[1 + rng.below(3) as usize])
        } else {
            (topics[1 + rng.below(5) as usize], form)
        };
        turns.push(Turn { topic, form, reply });
    }
    rng.shuffle(&mut turns);
    let mut transcript = String::new();
    let mut responses = Vec::new();
    for t in turns {
        let Turn {
            topic: tpc,
            form: fmt,
            reply,
        } = t;
        let request = if rng.below(2) == 0 {
            format!("User : write a {fmt} about {tpc} . ")
        } else {
            format!("User : for {tpc} , give me a {fmt} . ")
        };
        let response = if rng.below(2) == 0 {
            format!("Assistant : the word is {reply} . ")
        } else {
            format!("Assistant : I choose {reply} . ")
        };
        transcript.push_str(&request);
        transcript.push_str(&response);
        if tpc == topic && fmt == form {
            responses.push(reply);
        }
    }
    let mut prefix = vec![1];
    prefix.extend(text(ids, &transcript)?);
    if responses.len() != ORDINALS.len() || !seen.insert(prefix.clone()) {
        return Err("incorrect matching turn count or repeated transcript".into());
    }
    let template = rng.below(2);
    ORDINALS.into_iter().zip(responses).map(|(ordinal, reply)| {
        let question = if template == 0 {
            format!("User : repeat the {ordinal} {form} about {topic} . Assistant :")
        } else {
            format!("User : which word was requested for the {ordinal} {form} about {topic} . Assistant :")
        };
        let mut prompt = prefix.clone();
        prompt.extend(text(ids, &question)?);
        Ok(Example::generation(NAME, prompt, vec![ids[reply]]))
    }).collect()
}
pub(in crate::stress) fn data() -> Result<(Value, Task)> {
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, i as i64))
        .collect();
    let mut seen = HashSet::new();
    let mut sets = BTreeMap::new();
    let mut fingerprints = BTreeMap::new();
    for (name, seed, groups, noise) in [
        ("train", 2_391_211, 125, 4),
        ("validation", 2_501_119, 25, 4),
        ("test/fresh", 2_591_919, 25, 4),
        ("test/more-distractors", 2_739_099, 25, 12),
    ] {
        let mut rng = random::Random::new(seed);
        let mut rows = Vec::with_capacity(groups * 8);
        for _ in 0..groups {
            rows.extend(examples(&ids, &mut rng, noise, &mut seen)?);
        }
        if name == "train" {
            rng.shuffle(&mut rows);
        }
        let payload: Vec<_> = rows.iter().map(|r| (&r.prompt, &r.answer)).collect();
        fingerprints.insert(name, hash(&serde_json::to_vec(&payload)?));
        sets.insert(name.to_owned(), rows);
    }
    let case = json!({"width":64,"heads":2,"layers":2,"batch":16,"max_updates":3000,"observe":50});
    let digest = hash(&serde_json::to_vec(&(
        NAME,
        &words,
        &fingerprints,
        &case,
        "validation",
    ))?);
    let row = json!({"name":NAME,"tier":"synthetic-study","catalog_sha256":digest,"case":case,"selection":"validation",
        "train_examples":sets["train"].len(),"heldout_examples":sets["validation"].len(),"split_sha256":fingerprints});
    Ok((
        row,
        Task {
            vocab: words.len(),
            selection: "validation".into(),
            splits: sets,
        },
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn held_out_dialogues_are_disjoint_and_each_query_changes_only_one_ordinal_word() {
        let (_, task) = data().unwrap();
        let words = vocabulary();
        let mut transcripts = HashSet::new();
        for (split, rows) in &task.splits {
            let mut hist = [0; 8];
            for r in rows {
                let prompt: Vec<_> = r.prompt.iter().map(|&v| words[v as usize]).collect();
                let q = prompt.iter().rposition(|&s| s == "User").unwrap();
                let question = &prompt[q..];
                let ordinal = ORDINALS.iter().position(|s| question.contains(s)).unwrap();
                hist[ordinal] += 1;
                let active_form = question.iter().find(|s| FORMS.contains(s)).unwrap();
                let active_topic = question.iter().find(|s| TOPICS.contains(s)).unwrap();
                let mut expected = Vec::new();
                for turn in prompt[1..q]
                    .split(|&s| s == "User")
                    .filter(|a| !a.is_empty())
                {
                    if turn.contains(active_form) && turn.contains(active_topic) {
                        expected.push(
                            turn.iter()
                                .find(|w| VALUES.split_whitespace().any(|s| s == **w))
                                .unwrap(),
                        );
                    }
                }
                assert_eq!(expected.len(), 8);
                assert_eq!(*expected[ordinal], words[r.answer[0] as usize]);
                assert_eq!(r.tokens, r.prompt);
                assert_eq!(r.targets.iter().filter(|&&t| t != -100).count(), 1);
                assert_eq!(r.targets.last(), r.answer.first());
                if ordinal == 0 {
                    assert!(transcripts.insert(r.prompt[..q].to_vec()));
                }
            }
            assert_eq!(hist, [rows.len() / 8; 8]);
            if split == "train" {
                continue;
            }
            let (groups, remainder) = rows.as_chunks::<8>();
            assert!(remainder.is_empty());
            for group in groups {
                assert_eq!(
                    group
                        .iter()
                        .map(|r| r.answer[0])
                        .collect::<BTreeSet<_>>()
                        .len(),
                    8
                );
                for (i, r) in group.iter().enumerate() {
                    assert_eq!(r.prompt.len(), group[0].prompt.len());
                    let diffs = group[0]
                        .prompt
                        .iter()
                        .zip(&r.prompt)
                        .filter(|(a, b)| a != b)
                        .count();
                    assert_eq!(diffs, usize::from(i != 0));
                }
            }
        }
        assert_eq!(transcripts.len(), 200);
    }
}
