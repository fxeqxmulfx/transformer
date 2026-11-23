//! Consecutive pieces of one fingerprinted exact-sequence transfer split.
//! ProbeRows.correct proves that the counts add over a complete partition.
use super::*;
use std::ops::Range;

pub fn parse(args: &[String]) -> Result<(ProbeSet, Option<Range<usize>>)> {
    let (probe, range) = match args {
        [probe @ .., marker, a, b] if marker == "rows" => (probe, Some(a.parse()?..b.parse()?)),
        probe => (probe, None),
    };
    Ok((
        ProbeSet::parse(probe)?.ok_or("target-probes requires a transfer probe")?,
        range,
    ))
}

pub(super) fn restrict(
    task: &mut Task,
    hashes: &BTreeMap<String, String>,
    range: Range<usize>,
) -> Result<Value> {
    let mut keys = hashes.keys();
    let key = keys
        .next()
        .ok_or("row ranges require one fingerprinted transfer split")?;
    if keys.next().is_some() {
        return Err("ambiguous transfer split".into());
    }
    let rows = task
        .splits
        .get_mut(key)
        .ok_or("missing fingerprinted split")?;
    let size = rows.len();
    if range.is_empty() || range.end > size {
        return Err("transfer row interval is empty or outside the source split".into());
    }
    rows.truncate(range.end);
    rows.drain(..range.start);
    Ok(json!({"split":key,"start":range.start,"end":range.end,"source_rows":size,"batch":1}))
}

struct Part {
    template: Value,
    split: String,
    start: u64,
    end: u64,
    size: u64,
    counts: [u64; 2],
}

impl Part {
    fn read(mut sample: Value) -> Result<Self> {
        let info = sample
            .as_object_mut()
            .ok_or("invalid score")?
            .remove("probe_rows")
            .ok_or("missing row interval")?;
        let split = info["split"].as_str().ok_or("missing split")?.to_owned();
        let number = |key: &str| info[key].as_u64().ok_or("missing row index");
        let (start, end, size) = (number("start")?, number("end")?, number("source_rows")?);
        if info["batch"] != 1
            || start >= end
            || end > size
            || sample["probe_fingerprints"][&split].as_str().is_none()
        {
            return Err("invalid interval or fingerprint".into());
        }
        let mut counts = [0; 2];
        for (i, stage) in ["first", "final"].into_iter().enumerate() {
            if stage == "first" && sample[stage].is_null() {
                continue;
            }
            if sample[stage]["weights_sha256"].as_str().is_none()
                || sample[stage]["step"].as_u64().is_none()
            {
                return Err("missing checkpoint identity".into());
            }
            let score = sample[stage]["scores"]
                .as_object_mut()
                .and_then(|v| v.remove(&split))
                .ok_or("missing interval count")?;
            let correct = score["correct"].as_u64().ok_or("missing exact count")?;
            if score != json!({"correct":correct,"total":end-start}) || correct > end - start {
                return Err("interval score disagrees with its size or metric".into());
            }
            counts[i] = correct;
        }
        Ok(Self {
            template: sample,
            split,
            start,
            end,
            size,
            counts,
        })
    }
}

fn join(mut rows: Vec<Part>) -> Result<Value> {
    rows.sort_by_key(|r| r.start);
    let first = rows
        .first()
        .ok_or("supply the score files for one complete split")?;
    let mut end = 0;
    let mut counts = [0u64; 2];
    for part in &rows {
        if part.start != end
            || part.template != first.template
            || part.size != first.size
            || part.split != first.split
        {
            return Err(
                "interval gap, overlap, or incompatible checkpoint, scorer or source".into(),
            );
        }
        for (total, n) in counts.iter_mut().zip(part.counts) {
            *total += n;
        }
        end = part.end;
    }
    if end != first.size {
        return Err("missing final source rows".into());
    }
    let mut result = first.template.clone();
    for (stage, correct) in ["first", "final"].into_iter().zip(counts) {
        if !result[stage].is_null() {
            result[stage]["scores"][&first.split] = json!({"correct":correct,"total":end});
        }
    }
    Ok(result)
}

pub fn join_files(files: &[String]) -> Result<()> {
    let mut rows = Vec::with_capacity(files.len());
    for file in files {
        rows.push(Part::read(serde_json::from_reader(fs::File::open(file)?)?)?);
    }
    println!("{}", join(rows)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_one_complete_source_and_checkpoint_partition_can_be_joined() {
        let sample = |lo: usize, hi: usize, k: usize| {
            json!({"protocol":{"seed":0},"first":null,
            "final":{"weights_sha256":"w","step":550,"scores":{"test":{"correct":4,"total":5},"test/long":{"correct":k,"total":hi-lo}}},
            "probe_fingerprints":{"test/long":"abc"},"probe_rows":{"split":"test/long","start":lo,"end":hi,"source_rows":5,"batch":1}})
        };
        let a = sample(0, 3, 2);
        let b = sample(3, 5, 1);
        let assemble = |rows: Vec<Value>| -> Result<Value> {
            join(rows.into_iter().map(Part::read).collect::<Result<_>>()?)
        };
        let whole = assemble(vec![b.clone(), a.clone()]).unwrap();
        assert!(whole.get("probe_rows").is_none() && whole["first"].is_null());
        assert_eq!(
            whole["final"]["scores"]["test/long"],
            json!({"correct":3,"total":5})
        );
        assert_eq!(
            whole["final"]["scores"]["test"],
            json!({"correct":4,"total":5})
        );
        for parts in [
            vec![],
            vec![a.clone()],
            vec![b.clone()],
            vec![a.clone(), a.clone(), b.clone()],
            vec![a.clone(), sample(4, 5, 1)],
        ] {
            assert!(assemble(parts).is_err());
        }
        for path in [
            "/protocol/seed",
            "/final/weights_sha256",
            "/probe_fingerprints/test~1long",
            "/final/scores/test/correct",
        ] {
            let mut other = b.clone();
            *other.pointer_mut(path).unwrap() = json!("changed");
            assert!(assemble(vec![a.clone(), other]).is_err());
        }
        let mut bad = b;
        bad["final"]["scores"]["test/long"]["total"] = json!(5);
        assert!(Part::read(bad).is_err());
    }
}
