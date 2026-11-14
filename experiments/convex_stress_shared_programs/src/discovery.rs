//! Executable CM52 filtering. Every rejected program retains its first
//! contradiction; caps and an empty compatible set are recorded explicitly.
use crate::{program::Program, *};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub program: Program,
    pub observations: usize,
    pub positions: u64,
    pub conflict: Option<(usize, usize, bool)>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    pub input: Input,
    pub attempts: Vec<Attempt>,
    pub retained: Vec<Program>,
    pub positions: u64,
    pub observation_visits: u64,
    pub search_forecast: u64,
    pub buffer_forecast: u64,
    pub error: Option<String>,
}

pub fn validate(input: &Input) -> Result<(Vec<Program>, u64, u64)> {
    let inherited = sequences::complex::Request {
        input: input.clone(),
        policy: Default::default(),
    }
    .validate()?;
    let programs = program::catalogue(input.vocabulary, input.context)?;
    let tokens = input
        .observations
        .iter()
        .map(|s| s.tokens.len() as u64)
        .sum::<u64>();
    let forecast = 4 * programs.len() as u64 * tokens;
    if forecast > 2_000_000_000 {
        return Err("program search exceeds2B position forecast".into());
    }
    let bytes = inherited
        + 256 * programs.len() as u64
        + 128 * 1024 * input.observations.len() as u64
        + 1024 * (input.vocabulary as u64).pow(2);
    if bytes > 56 * 1024 * 1024 * 1024 {
        return Err("program binding exceeds56GiB buffer forecast".into());
    }
    Ok((programs, forecast, bytes))
}

pub fn run(input: Input) -> Result<Report> {
    let (programs, forecast, bytes) = validate(&input)?;
    let mut out = Report {
        input,
        attempts: vec![],
        retained: vec![],
        positions: 0,
        observation_visits: 0,
        search_forecast: forecast,
        buffer_forecast: bytes,
        error: None,
    };
    for program in programs {
        let mut attempt = Attempt {
            program,
            observations: 0,
            positions: 0,
            conflict: None,
        };
        for (index, observation) in out.input.observations.iter().enumerate() {
            let value = program.run(&observation.tokens);
            attempt.observations += 1;
            attempt.positions += value.positions;
            if value.state != observation.label {
                attempt.conflict = Some((index, value.state, value.missing));
                break;
            }
        }
        out.positions += attempt.positions;
        out.observation_visits += attempt.observations as u64;
        if attempt.conflict.is_none() {
            out.retained.push(program);
        }
        out.attempts.push(attempt);
        if out.retained.len() > 1024 {
            out.error = Some("retained program cap1024".into());
            break;
        }
    }
    if out.retained.is_empty() {
        out.error = Some("no training-compatible copy program".into());
    }
    Ok(out)
}

pub fn check(input: &Input, out: &Report) -> Result<()> {
    if out.input != *input || run(input.clone())? != *out {
        return Err("input-only program filter, conflicts, caps or work changed".into());
    }
    for attempt in &out.attempts {
        if let Some((index, state, missing)) = attempt.conflict {
            let observation = &input.observations[index];
            if attempt.program.interpret(&observation.tokens) != (state, missing)
                || state == observation.label
            {
                return Err("independent program contradiction differs".into());
            }
        } else {
            for observation in &input.observations {
                if attempt.program.interpret(&observation.tokens).0 != observation.label {
                    return Err("retained program fails independent original labels".into());
                }
            }
        }
    }
    Ok(())
}
