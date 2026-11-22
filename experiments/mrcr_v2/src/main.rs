//! Run the official eight-needle MRCR V2 stress evaluation on saved text completions.
mod metric;

use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Seek, Write},
    path::Path,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SOURCES: &str = include_str!("../sources.tsv");

#[derive(Deserialize)]
struct Example {
    queries: String,
    answer: String,
    num_relevant: u32,
    context_len: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Prediction {
    id: String,
    response: String,
}

fn predictions(reader: impl BufRead) -> Result<BTreeMap<String, String>> {
    let mut values = BTreeMap::new();
    for line in reader.lines() {
        let record: Prediction = serde_json::from_str(&line?)?;
        if values.insert(record.id.clone(), record.response).is_some() {
            return Err(format!("duplicate prediction for {}", record.id).into());
        }
    }
    Ok(values)
}

fn verified(split: &str, path: &Path) -> Result<BufReader<File>> {
    let row = SOURCES
        .lines()
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .find(|columns| columns.first() == Some(&split))
        .ok_or("unknown MRCR split")?;
    let mut reader = BufReader::with_capacity(128 * 1024, File::open(path)?);
    if reader.get_ref().metadata()?.len() != row[3].parse::<u64>()? {
        return Err("incorrect MRCR file size".into());
    }
    let mut hash = Sha256::new();
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            break;
        }
        hash.update(chunk);
        let n = chunk.len();
        reader.consume(n);
    }
    let digest: String = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if digest != row[4] {
        return Err("incorrect MRCR content digest".into());
    }
    reader.rewind()?;
    Ok(reader)
}

enum Action {
    Verify,
    Prompts,
    Score {
        predictions: BTreeMap<String, String>,
        model: String,
        tools: bool,
    },
}

fn evaluate(
    reader: impl io::Read,
    split: &str,
    mut action: Action,
    mut out: impl Write,
) -> Result<()> {
    let mut csv = csv::Reader::from_reader(reader);
    let (mut count, mut perfect, mut sum) = (0, 0, 0.);
    let (mut shortest, mut longest) = (usize::MAX, 0);
    for parsed in csv.deserialize::<Example>() {
        let row = parsed?;
        if row.num_relevant != 8
            || row.queries.is_empty()
            || row.answer.chars().count() < 12
            || row.context_len == 0
        {
            return Err(format!("invalid eight-needle example at row {count}").into());
        }
        let id = format!("{split}/{count}");
        shortest = shortest.min(row.context_len);
        longest = longest.max(row.context_len);
        match &mut action {
            Action::Prompts => {
                serde_json::to_writer(&mut out, &json!({"id":id, "prompt":row.queries}))?;
                writeln!(out)?;
            }
            Action::Score { predictions, .. } => {
                let response = predictions
                    .remove(&id)
                    .ok_or_else(|| format!("missing prediction for {id}"))?;
                let value = metric::grade(&response, &row.answer);
                sum += value;
                perfect += usize::from(value == 1.);
            }
            Action::Verify => {}
        }
        count += 1;
    }
    if count == 0 {
        return Err("empty MRCR split".into());
    }
    let mut report = json!({"case":"mrcr-v2-8needle","split":split,"rows":count,
        "min_context_tokens":shortest,"max_context_tokens":longest});
    match action {
        Action::Prompts => return Ok(out.flush()?),
        Action::Verify => {}
        Action::Score {
            predictions,
            model,
            tools,
        } => {
            if let Some(id) = predictions.keys().next() {
                return Err(format!("unexpected prediction for {id}").into());
            }
            report["model"] = model.into();
            report["tools"] = tools.into();
            report["mean_match_ratio"] = (sum / f64::from(count)).into();
            report["perfect"] = perfect.into();
        }
    }
    writeln!(out, "{report}")?;
    out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (split, dataset, action) = match args.as_slice() {
        [cmd, split, file] if cmd == "verify" => (split, file, Action::Verify),
        [cmd, split, file] if cmd == "prompts" => (split, file, Action::Prompts),
        [cmd, split, file, completions, model, tools] if cmd == "score" && !model.trim().is_empty() => {
            let tools = match tools.as_str() {
                "tools" => true, "no-tools" => false, _ => return Err("choose tools or no-tools".into()),
            };
            let predictions = predictions(BufReader::new(File::open(completions)?))?;
            (split, file, Action::Score { predictions, model: model.clone(), tools })
        }
        _ => return Err("usage: mrcr-v2 {verify|prompts} SPLIT CSV | score SPLIT CSV PREDICTIONS.jsonl MODEL {tools|no-tools}".into())
    };
    evaluate(
        verified(split, Path::new(dataset))?,
        split,
        action,
        BufWriter::new(io::stdout().lock()),
    )
}

#[cfg(test)]
mod tests;
