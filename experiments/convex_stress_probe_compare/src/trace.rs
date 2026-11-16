//! Every call records SHA256 of its exact visible prefix and ALL binary64
//! class logits. Independent replay never calls a learned head or decoder.
use crate::{resources::Guard, *};
use probes::{
    boundary::Predictor,
    dataset::{Dataset, Split},
};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::Path,
};
const MAGIC: &[u8; 8] = b"PRBLOG01";
fn prefix_hash(prefix: &[i64]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update((prefix.len() as u64).to_le_bytes());
    for token in prefix {
        hash.update(token.to_le_bytes());
    }
    hash.finalize().into()
}
pub(crate) fn maximum_calls(split: &Split, name: &str) -> Result<u64> {
    let mut count = 0u64;
    for row in &split.records {
        let n = row.targets.iter().filter(|&&x| x != -100).count() as u64;
        count = count.checked_add(n).ok_or("trace call bound overflow")?;
        if !name.starts_with("train") && !row.prompt.is_empty() {
            count = count
                .checked_add(row.generation_limit as u64)
                .ok_or("trace call bound overflow")?;
        }
    }
    if count == 0 || count > 2_097_152 {
        return Err("trace original call count exceeds admission".into());
    }
    Ok(count)
}
pub(crate) fn length(classes: usize, calls: u64) -> Result<u64> {
    if !(2..=1024).contains(&classes) || calls > 2_097_152 {
        return Err("invalid raw trace shape".into());
    }
    16u64
        .checked_add(
            calls
                .checked_mul(32 + 8 * classes as u64)
                .ok_or("trace length overflow")?,
        )
        .ok_or_else(|| "trace length overflow".into())
}
pub struct Recorder<'a, P> {
    model: P,
    file: io::Writer<'a>,
    classes: usize,
    calls: u64,
    maximum: u64,
}
impl<'a, P: Predictor> Recorder<'a, P> {
    pub fn new(
        model: P,
        path: &Path,
        classes: usize,
        maximum: u64,
        guard: &'a Guard,
    ) -> Result<Self> {
        let mut file = io::Writer::new(path, length(classes, maximum)?, guard)?;
        file.write_all(MAGIC)?;
        file.write_all(&(classes as u64).to_le_bytes())?;
        Ok(Self {
            model,
            file,
            classes,
            calls: 0,
            maximum,
        })
    }
    pub fn finish(self) -> Result<(io::Artifact, u64)> {
        let artifact = self.file.finish()?;
        if artifact.bytes != length(self.classes, self.calls)? {
            return Err("raw trace length differs from original calls".into());
        }
        Ok((artifact, self.calls))
    }
}
impl<P: Predictor> Predictor for Recorder<'_, P> {
    fn logits(&mut self, prefix: &[i64]) -> probes::Result<Vec<f64>> {
        if self.calls >= self.maximum {
            return Err("original score exceeded frozen trace bound".into());
        }
        let values = self.model.logits(prefix)?;
        if values.len() != self.classes || values.iter().any(|x| !x.is_finite()) {
            return Err("invalid complete raw logit row".into());
        }
        self.file.write_all(&prefix_hash(prefix))?;
        for x in &values {
            self.file.write_all(&x.to_bits().to_le_bytes())?;
        }
        self.calls += 1;
        Ok(values)
    }
}
pub struct Replay {
    file: BufReader<File>,
    classes: usize,
    remaining: u64,
}
impl Replay {
    pub fn new(path: &Path, classes: usize, calls: u64, expected: &io::Artifact) -> Result<Self> {
        if io::artifact(path)? != *expected || expected.bytes != length(classes, calls)? {
            return Err("complete raw trace changed".into());
        }
        let mut file = BufReader::with_capacity(262144, File::open(path)?);
        let mut header = [0; 16];
        file.read_exact(&mut header)?;
        if &header[..8] != MAGIC || u64::from_le_bytes(header[8..].try_into()?) != classes as u64 {
            return Err("raw trace header changed".into());
        }
        Ok(Self {
            file,
            classes,
            remaining: calls,
        })
    }
    pub fn finish(mut self) -> Result<()> {
        let mut extra = [0; 1];
        if self.remaining != 0 || self.file.read(&mut extra)? != 0 {
            return Err("raw replay omitted or added original calls".into());
        }
        Ok(())
    }
}
impl Predictor for Replay {
    fn logits(&mut self, prefix: &[i64]) -> probes::Result<Vec<f64>> {
        if self.remaining == 0 {
            return Err("raw replay requested an extra call".into());
        }
        let mut hash = [0; 32];
        self.file.read_exact(&mut hash)?;
        if hash != prefix_hash(prefix) {
            return Err("raw replay visible prefix or original order changed".into());
        }
        let mut out = Vec::with_capacity(self.classes);
        for _ in 0..self.classes {
            let mut word = [0; 8];
            self.file.read_exact(&mut word)?;
            let x = f64::from_bits(u64::from_le_bytes(word));
            if !x.is_finite() {
                return Err("nonfinite raw replay logit".into());
            }
            out.push(x);
        }
        self.remaining -= 1;
        Ok(out)
    }
}
pub(crate) fn record(
    data: &Dataset,
    split: &str,
    model: impl Predictor,
    path: &Path,
    guard: &Guard,
) -> Result<crate::schema::SplitScore> {
    let maximum = maximum_calls(
        data.splits.get(split).ok_or("missing original split")?,
        split,
    )?;
    let mut recorder = Recorder::new(model, path, data.vocabulary, maximum, guard)?;
    let score = probe(probes::scoring::score(data, split, &mut recorder))?;
    let (trace, calls) = recorder.finish()?;
    Ok(crate::schema::SplitScore {
        split: split.into(),
        trace,
        calls,
        score,
    })
}
