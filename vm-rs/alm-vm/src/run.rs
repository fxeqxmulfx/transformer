//! The generation loop, and what the original reports about it.

use alm_model::{Alm, Backend, KvCache};

/// What one program run produced.
pub struct Run {
    /// Prompt and generation together, as the original counts them.
    pub ids: Vec<usize>,
    /// Tokens the trace calls operations: any commit, or a taken branch.
    pub ops: usize,
    pub seconds: f64,
    /// Whether generation ended on the stop token rather than the budget.
    pub stopped: bool,
}

/// Greedy generation, one token at a time, exactly as `generate_with_cache`
/// and the C++ driver do it: every position is pushed through the stack, and
/// only the last one is decoded.
pub fn generate(
    model: &Alm<Backend>,
    cache: &mut KvCache,
    prompt: &[usize],
    max_new: usize,
    trace_every: usize,
) -> Run {
    let started = std::time::Instant::now();
    let mut ids = prompt.to_vec();
    ids.reserve(max_new);
    let stop = model.shapes.stop_token;
    let mut stopped = false;

    for pos in 0..prompt.len() + max_new {
        let x = model.forward(ids[pos], pos, cache);
        if pos + 1 == ids.len() {
            let next = model.decode(&x);
            ids.push(next);
            let gen = pos + 1 - prompt.len();
            if trace_every > 0 && (gen.is_multiple_of(trace_every) || next == stop) {
                let t = started.elapsed().as_secs_f64();
                let rate = if t > 0.0 { gen as f64 / t } else { 0.0 };
                eprintln!("[{gen:7} {t:7.3}s {rate:6.0} tok/s] {}", model.tokens[next]);
            }
            if next == stop {
                stopped = true;
                break;
            }
        }
    }

    let ops = ids
        .iter()
        .filter(|&&i| {
            let t = &model.tokens[i];
            t.contains("commit") || t == "branch_taken"
        })
        .count();
    Run { ids, ops, seconds: started.elapsed().as_secs_f64(), stopped }
}

/// The bytes the program wrote: every `out(..)` token, its payload either one
/// literal character or a hex byte.
pub fn output_bytes(model: &Alm<Backend>, ids: &[usize]) -> Vec<u8> {
    let mut out = Vec::new();
    for &i in ids {
        let t = &model.tokens[i];
        let Some(body) = t.strip_prefix("out(").and_then(|b| b.strip_suffix(')')) else {
            continue;
        };
        let mut chars = body.chars();
        match (chars.next(), chars.next()) {
            (Some(c), None) => out.push(c as u8),
            _ => {
                if let Ok(b) = u8::from_str_radix(body, 16) {
                    out.push(b);
                }
            }
        }
    }
    out
}
