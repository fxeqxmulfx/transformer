//! Final-position votes at unit gain, projected against the entire tied
//! vocabulary. Inputs pass through the learned geometric dictionary.
use super::*;
pub(in crate::stress) mod aliases;
pub mod long;

/// Use the pinned DiffGPT base_ops::rms warp's sum order. The host
/// reciprocal square root can round differently from CUDA rsqrt.approx.
fn normalized(row: &[f32]) -> Vec<f32> {
    let mut lanes = [0.0f32; 32];
    for (j, &x) in row.iter().enumerate() {
        lanes[j % 32] += x * x;
    }
    for stride in [16, 8, 4, 2, 1] {
        for i in 0..stride {
            lanes[i] += lanes[i + stride];
        }
    }
    let scale = (lanes[0] / row.len() as f32 + f32::EPSILON).sqrt().recip();
    row.iter().map(|&x| scale * x).collect()
}

fn classify(embeddings: &[f32], hidden: &[f32]) -> (usize, bool) {
    let mut winner = (0, false);
    let mut highest = f32::NEG_INFINITY;
    for (id, word) in embeddings.chunks_exact(hidden.len()).enumerate() {
        let score = crate::slot_memory::dot(word, hidden);
        if score > highest {
            highest = score;
            winner = (id, false);
        } else if score == highest {
            winner.1 = true;
        }
    }
    winner
}

struct Geometry<'a> {
    embeddings: &'a [f32],
    values: Vec<Vec<f32>>,
    addresses: Vec<Option<usize>>,
    width: usize,
}

impl<'a> Geometry<'a> {
    fn new(cfg: &Config, weights: &'a [f32]) -> Self {
        let layout = crate::slot_memory::Layout::new(cfg);
        let store = &weights[architecture::memory_start(cfg)..];
        let mut routing = crate::slot_memory::Routing::new(cfg.head);
        routing.geometry(&layout, store);
        let embeddings = &weights[..cfg.vocab * cfg.width];
        let values: Vec<_> = embeddings.chunks_exact(cfg.width).map(normalized).collect();
        let addresses = values
            .iter()
            .map(|v| routing.nearest(&layout, store, v))
            .collect();
        Self {
            embeddings,
            values,
            addresses,
            width: cfg.width,
        }
    }

    fn answer(&self, prompt: &[i64], passes: usize, table: &[f32], ordered: bool) -> (usize, bool) {
        self.answers(prompt, &[passes], table, ordered, None)[0]
    }

    fn answers(
        &self,
        prompt: &[i64],
        budgets: &[usize],
        table: &[f32],
        ordered: bool,
        agreement_power: Option<u8>,
    ) -> Vec<(usize, bool)> {
        let keys: Vec<_> = prompt
            .iter()
            .map(|&token| self.addresses[token as usize])
            .collect();
        crate::typed_lookup::final_vote_budgets(&keys, budgets, table, ordered, agreement_power)
            .into_iter()
            .map(|votes| self.project(prompt, &votes))
            .collect()
    }

    fn project(&self, prompt: &[i64], votes: &[f32]) -> (usize, bool) {
        let mut hidden = vec![0.; self.width];
        for (&v, &token) in votes.iter().zip(prompt) {
            if v != 0. {
                for (dst, &x) in hidden.iter_mut().zip(&self.values[token as usize]) {
                    *dst += v * x;
                }
            }
        }
        classify(self.embeddings, &hidden)
    }
}

pub(in crate::stress) fn budget_predictions(
    cfg: &Config,
    weights: &[f32],
    examples: &[Example],
    budgets: &[usize],
) -> Result<Vec<Vec<Option<i64>>>> {
    let geometry = Geometry::new(cfg, weights);
    let path = cfg
        .head
        .peer_read()
        .and_then(|r| r.path)
        .ok_or("no discrete path branch")?;
    let table = crate::typed_lookup::inference_table(
        &weights[coefficients::trained_range(cfg)?],
        path,
    );
    let span = examples
        .len()
        .div_ceil(crate::typed_lookup::worker_limit())
        .max(1);
    let mut rows = vec![Vec::new(); examples.len()];
    std::thread::scope(|s| {
        for (inputs, outputs) in examples.chunks(span).zip(rows.chunks_mut(span)) {
            s.spawn(|| {
                for (input, output) in inputs.iter().zip(outputs) {
                    *output = geometry.answers(
                        &input.prompt,
                        budgets,
                        &table,
                        path.ordered,
                        path.agreement.then_some(path.gain_power),
                    );
                }
            });
        }
    });
    let mut predictions = vec![Vec::with_capacity(examples.len()); budgets.len()];
    for row in rows {
        for (out, (id, _)) in predictions.iter_mut().zip(row) {
            out.push(Some(id as i64));
        }
    }
    Ok(predictions)
}

pub fn run(seed: u32, out: &Path, passes: usize, selection: &str, gain: &str) -> Result<()> {
    if passes == 0 {
        return Err("an isolated read requires a positive pass count".into());
    }
    let (row, task) = data()?;
    let cfg = task_config(&row, &task, true)?;
    let (mut weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let digest = weight_digest(&weights);
    let range = coefficients::table_range(&cfg)?;
    let params = coefficients::transform(&mut weights[range.clone()], selection, gain)?;
    let table = &weights[range];
    let geometry = Geometry::new(&cfg, &weights);
    let ordered = cfg
        .head
        .peer_read()
        .and_then(|r| r.path)
        .is_some_and(|p| p.ordered);
    let names = vocabulary();
    let mut scores = BTreeMap::new();
    for (split, rows) in &task.splits {
        let mut correct = 0;
        let mut ties = 0;
        let mut fixture = None;
        for example in rows {
            let (predicted, tied) = geometry.answer(&example.prompt, passes, table, ordered);
            correct += usize::from(example.answer.as_slice() == [predicted as i64]);
            ties += usize::from(tied);
            if split == "test/mira" {
                fixture = Some(&names[predicted]);
            }
        }
        let mut score = json!({"correct":correct,"total":rows.len(),"max_logit_ties":ties});
        if let Some(word) = fixture {
            score["prediction"] = json!(word);
        }
        scores.insert(split, score);
    }
    println!(
        "{}",
        json!({"model":MODEL,"seed":seed,"step":point["step"],"weights_sha256":digest,
        "protocol_sha256":row["catalog_sha256"],"training_kernite":KERNITE_VERSION,"passes":passes,
        "evaluation":"isolated unit-gain read with tied full-vocabulary projection on CPU",
        "coefficients":params,"scores":scores})
    );
    Ok(())
}
