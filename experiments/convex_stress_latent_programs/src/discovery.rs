//! CM53 dictionary fitting. Every compatible program survives. Retain the
//! first two conflicting original record IDs; unobserved rows stay None.
use crate::{program::Program, *};
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Learned {
    pub program: Program,
    pub decoder: Vec<Option<usize>>,
    pub source_rows: Vec<Option<usize>>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub program: Program,
    pub observations: usize,
    pub positions: u64,
    pub conflict: Option<(usize, usize, usize, usize, usize)>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    pub input: Input,
    pub attempts: Vec<Attempt>,
    pub retained: Vec<Learned>,
    pub positions: u64,
    pub search_forecast: u64,
    pub buffer_forecast: u64,
    pub error: Option<String>,
}
pub fn run(input: Input) -> Result<Report> {
    let inherited = sequences::complex::Request {
        input: input.clone(),
        policy: Default::default(),
    }
    .validate()?;
    let programs = program::catalogue(input.vocabulary)?;
    // Includes the independent prefix-sum interpreter's quadratic work.
    let tokens: u64 = input
        .observations
        .iter()
        .map(|s| (s.tokens.len() as u64).pow(2) + 8 * s.tokens.len() as u64)
        .sum();
    let forecast = programs.len() as u64 * tokens;
    if forecast > WORK_LIMIT {
        return Err("latent search exceeds16B scalar-work forecast".into());
    }
    let bytes = inherited
        + 1024 * programs.len() as u64
        + 128 * input.observations.len() as u64 * 128
        + 1024 * 128 * STATES as u64 * input.vocabulary as u64;
    if bytes > 56 * 1024 * 1024 * 1024 {
        return Err("latent binding exceeds56GiB buffer forecast".into());
    }
    let mut out = Report {
        input,
        attempts: vec![],
        retained: vec![],
        positions: 0,
        search_forecast: forecast,
        buffer_forecast: bytes,
        error: None,
    };
    for program in programs {
        let mut learned = Learned {
            program,
            decoder: vec![None; STATES],
            source_rows: vec![None; STATES],
        };
        let mut attempt = Attempt {
            program,
            observations: 0,
            positions: 0,
            conflict: None,
        };
        for (i, row) in out.input.observations.iter().enumerate() {
            let v = program.run(&row.tokens);
            attempt.observations += 1;
            attempt.positions += v.positions;
            if let Some(label) = learned.decoder[v.state] {
                if label != row.label {
                    attempt.conflict = Some((
                        learned.source_rows[v.state].unwrap(),
                        i,
                        v.state,
                        label,
                        row.label,
                    ));
                    break;
                }
            } else {
                learned.decoder[v.state] = Some(row.label);
                learned.source_rows[v.state] = Some(i);
            }
        }
        out.positions += attempt.positions;
        if attempt.conflict.is_none() {
            out.retained.push(learned);
        }
        out.attempts.push(attempt);
        if out.retained.len() > 128 {
            out.error = Some("retained latent program cap128".into());
            break;
        }
    }
    if out.retained.is_empty() {
        out.error = Some("no training-compatible latent summary".into());
    }
    Ok(out)
}
pub fn check(input: &Input, out: &Report) -> Result<()> {
    if out.input != *input || run(input.clone())? != *out {
        return Err("latent input/filter/dictionary/work replay differs".into());
    }
    for attempt in &out.attempts {
        if let Some((first, second, state, a, b)) = attempt.conflict
            && (first >= second
                || a == b
                || input.observations[first].label != a
                || input.observations[second].label != b
                || [first, second]
                    .iter()
                    .any(|&i| attempt.program.interpret(&input.observations[i].tokens) != state))
        {
            return Err("independent latent conflict witness differs".into());
        }
    }
    for learned in &out.retained {
        for (i, row) in input.observations.iter().enumerate() {
            let state = learned.program.interpret(&row.tokens);
            if learned.decoder[state] != Some(row.label)
                || learned.source_rows[state].is_none_or(|s| s > i)
            {
                return Err("independent learned dictionary differs".into());
            }
        }
    }
    Ok(())
}
