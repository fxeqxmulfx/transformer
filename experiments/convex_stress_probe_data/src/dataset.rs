//! All six original probes, including their complete numeric teachers and masks.
//! Compare every source fingerprint before returning any dataset to the driver.
use crate::{
    Result, data, native,
    wire::{self, Record},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    pub records: Vec<Record>,
    pub teacher_words: Option<Vec<Vec<u64>>>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub name: String,
    pub vocabulary: usize,
    pub selection: String,
    pub numeric_step: Option<f64>,
    pub numeric_offset: Option<i64>,
    pub splits: BTreeMap<String, Split>,
}
impl Dataset {
    pub fn validate(&self, source: &Value) -> Result<()> {
        let fingerprints = source["history"]["data_fingerprints"]
            .as_object()
            .ok_or("source fingerprints")?;
        let task = source["case"]["data"]["task"]
            .as_str()
            .ok_or("source task")?;
        let (vocabulary, numeric) = match task {
            "routing-mixture" => (293, true),
            "branch-amplitudes" | "conditional-amplitudes" | "linear-transport" => (69, true),
            "order-discrimination" => (70, false),
            "key-value-binding" => (100, false),
            _ => return Err("unknown original probe".into()),
        };
        let selection = if task == "key-value-binding" {
            "validation/novel"
        } else {
            "validation"
        };
        if source["case"]["name"] != self.name
            || self.vocabulary != vocabulary
            || self.selection != selection
            || self.numeric_step.is_some() != numeric
            || self.numeric_offset.is_some() != numeric
            || self.splits.len() != fingerprints.len()
            || !self.splits.contains_key(&self.selection)
            || !self.splits.contains_key("train")
        {
            return Err("complete source split inventory changed".into());
        }
        for (name, split) in &self.splits {
            for row in &split.records {
                row.validate(self.vocabulary)?;
                // Dense fingerprints do not include this descriptive field.
                if task == "order-discrimination" && row.task != task {
                    return Err("dense source task metadata changed".into());
                }
            }
            let actual = if self.name == "gpt-probe-order-discrimination" {
                wire::tensor_hash(&split.records)
            } else if self.name == "gpt-probe-key-value-binding" {
                wire::binding_hash(name, &split.records)
            } else {
                wire::examples_hash(&split.records)?
            };
            if fingerprints.get(name) != Some(&json!(actual)) {
                return Err(format!(
                    "original {}/{} fingerprint differs: {actual}",
                    self.name, name
                )
                .into());
            }
        }
        // Check the entire tokenized cohort before independent numeric teachers.
        for (name, split) in &self.splits {
            match (&split.teacher_words, self.numeric_step, self.numeric_offset) {
                (Some(words), Some(step), Some(offset)) => {
                    let actual_teacher = wire::teacher_hash(words)?;
                    if words.len() != split.records.len()
                        || !step.is_finite()
                        || step <= 0.
                        || source["history"]["design"]["numeric_grid"]["step"] != step
                        || source["history"]["design"]["numeric_grid"]["offset"] != offset
                        || source["history"]["design"]["numeric_grid"]["unquantized_teacher_sha256"]
                            [name]
                            != actual_teacher
                    {
                        return Err(format!(
                            "complete unquantized teacher differs for {}/{name}: actual={actual_teacher}, expected={}",
                            self.name,source["history"]["design"]["numeric_grid"]["unquantized_teacher_sha256"][name]
                        )
                        .into());
                    }
                    for (row, truth) in split.records.iter().zip(words) {
                        if truth.len() + 1 != row.answer.len()
                            || truth.iter().zip(&row.answer).any(|(&bits, &label)| {
                                (f64::from_bits(bits) / step).round_ties_even() as i64 + offset + 36
                                    != label
                            })
                        {
                            return Err("numeric teacher and original answer grid differ".into());
                        }
                    }
                }
                (None, None, None) if !numeric => {}
                _ => return Err("missing numeric teacher or original codec".into()),
            }
        }
        Ok(())
    }
    pub fn summary(&self) -> Result<Value> {
        let mut splits = BTreeMap::new();
        for (name, value) in &self.splits {
            splits.insert(
                name,
                wire::summary(&value.records, value.teacher_words.as_deref())?,
            );
        }
        Ok(
            json!({"name":self.name,"vocabulary":self.vocabulary,"selection":self.selection,
            "numeric_step":self.numeric_step,"numeric_offset":self.numeric_offset,"splits":splits,
            "model_updates":0,"model_measurements":0}),
        )
    }
}
pub fn build(source: &Value, helper: &Path) -> Result<Dataset> {
    let name = source["case"]["name"].as_str().ok_or("source case name")?;
    let task = source["case"]["data"]["task"]
        .as_str()
        .ok_or("source task")?;
    if task == "key-value-binding" {
        let original: Value = serde_json::from_str(crate::ORIGINAL35)?;
        let row = original["tasks"]
            .as_array()
            .ok_or("original35 tasks")?
            .iter()
            .find(|r| r["name"] == name)
            .ok_or("missing old probe control")?;
        let value = data::build(row)?;
        let out = Dataset {
            name: name.into(),
            vocabulary: value.vocab,
            selection: value.selection,
            numeric_step: None,
            numeric_offset: None,
            splits: value
                .splits
                .into_iter()
                .map(|(name, rows)| {
                    (
                        name,
                        Split {
                            records: rows.iter().map(Record::from_original).collect(),
                            teacher_words: None,
                        },
                    )
                })
                .collect(),
        };
        out.validate(source)?;
        return Ok(out);
    }
    let numeric = task != "order-discrimination";
    let mut out = Dataset {
        name: name.into(),
        vocabulary: 0,
        selection: "validation".into(),
        numeric_step: numeric.then_some(if task == "routing-mixture" {
            1. / 32.
        } else {
            0.25
        }),
        numeric_offset: numeric.then_some(if task == "routing-mixture" { 128 } else { 16 }),
        splits: BTreeMap::new(),
    };
    let names = if numeric {
        &["train", "validation", "test"][..]
    } else {
        &["train", "validation"][..]
    };
    for &split in names {
        let raw = native::generate(helper, task, split)?;
        if out.vocabulary != 0 && out.vocabulary != raw.vocabulary {
            return Err("split vocabulary changed".into());
        }
        out.vocabulary = raw.vocabulary;
        let records = raw
            .inputs
            .iter()
            .zip(&raw.labels)
            .map(|(input, labels)| {
                if numeric {
                    Record::generation(task, input, labels)
                } else {
                    Record {
                        task: task.into(),
                        tokens: input.clone(),
                        targets: vec![-100, labels[0]],
                        changes: vec![false, false],
                        prompt: vec![],
                        answer: vec![],
                        generation_limit: 0,
                    }
                }
            })
            .collect();
        out.splits.insert(
            split.into(),
            Split {
                records,
                teacher_words: raw.teacher,
            },
        );
    }
    out.validate(source)?;
    Ok(out)
}
