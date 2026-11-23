//! Five different starting-key questions per set of unnumbered printed tables.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::stress::lookup_question) enum Relations {
    Shared,
    Rebound,
}
impl Relations {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "shared" => Ok(Self::Shared),
            "rebound" => Ok(Self::Rebound),
            _ => Err("choose shared or rebound table relations".into()),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Rebound => "rebound",
        }
    }
}

fn printed(q: &Question) -> String {
    let mut blocks = Vec::with_capacity(q.tables.len() + 1);
    for t in &q.tables {
        let rows = t
            .iter()
            .map(|(a, b)| format!("{a} → {b}"))
            .collect::<Vec<_>>()
            .join("   ");
        blocks.push(format!("Table\n{rows}"));
    }
    let options = LETTERS
        .iter()
        .zip(&q.choices)
        .map(|(l, v)| format!("{l}. {v}"))
        .collect::<Vec<_>>()
        .join("   ");
    blocks.push(format!(
        "Starting at {}, what do you get?\n{options}",
        q.start
    ));
    blocks.join("\n\n")
}

pub(in crate::stress::lookup_question) fn long(
    hops: usize,
    mode: Relations,
) -> Result<Vec<Example>> {
    let positions = hops
        .checked_mul(16)
        .and_then(|x| x.checked_add(24))
        .ok_or("sequence length overflow")?;
    if hops == 0 || positions > 8192 {
        return Err("supply a positive depth fitting the 8192-position scorer".into());
    }
    let words = vocabulary();
    let ids = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as i64))
        .collect();
    let mut rng =
        random::Random::new(1_104_211 + 5 * hops as u32 + u32::from(mode == Relations::Rebound));
    let mut out = Vec::with_capacity(100);
    let mut seen = HashSet::new();
    for _ in 0..20 {
        let mut names: Vec<_> = NAMES.split_whitespace().collect();
        rng.shuffle(&mut names);
        let successor: BTreeMap<_, _> = names
            .iter()
            .copied()
            .zip(names.iter().cycle().skip(1).copied())
            .collect();
        rng.shuffle(&mut names);
        let starts = names[..5].to_vec();
        let mut layer = starts.clone();
        let mut tables = Vec::with_capacity(hops);
        for _ in 0..hops {
            let next = match mode {
                Relations::Shared => layer.iter().map(|v| successor[v]).collect::<Vec<_>>(),
                Relations::Rebound => {
                    rng.shuffle(&mut names);
                    names[..5].to_vec()
                }
            };
            let mut table = layer
                .iter()
                .zip(&next)
                .map(|(&a, &b)| (a.to_owned(), b.to_owned()))
                .collect::<Table>();
            rng.shuffle(&mut table);
            tables.push(table);
            layer = next;
        }
        let mut options: Vec<_> = layer.iter().map(|v| (*v).to_owned()).collect();
        rng.shuffle(&mut options);
        let mut q = Question {
            tables,
            start: String::new(),
            choices: options,
        };
        for (start, &last) in starts.iter().zip(&layer) {
            q.start = (*start).to_owned();
            let expected = q
                .choices
                .iter()
                .position(|x| x == last)
                .ok_or("missing final value")?;
            if q.answer()? != expected {
                return Err("the independent table walk changed the lane endpoint".into());
            }
            let row = encode(&printed(&q), expected, &ids)?;
            if !seen.insert(row.prompt.clone()) {
                return Err("duplicate long lookup prompt".into());
            }
            out.push(row);
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_five_answers_share_the_same_context_and_vocabulary_at_forty_hops() {
        for mode in [Relations::Shared, Relations::Rebound] {
            let words = vocabulary();
            let rows = long(40, mode).unwrap();
            for set in rows.as_chunks::<5>().0 {
                let mut answers = BTreeSet::new();
                let mut starts = BTreeSet::new();
                let marker = words.iter().position(|w| w == "Starting").unwrap() as i64;
                let start = set[0].prompt.iter().position(|&i| i == marker).unwrap() + 2;
                for r in set {
                    assert_eq!(r.prompt[..start], set[0].prompt[..start]);
                    assert_eq!(r.prompt[start + 1..], set[0].prompt[start + 1..]);
                    assert!(answers.insert(r.answer[0]) && starts.insert(r.prompt[start]));
                    assert_eq!(r.targets.iter().filter(|&&v| v != -100).count(), 1);
                }
            }
        }
    }
}
