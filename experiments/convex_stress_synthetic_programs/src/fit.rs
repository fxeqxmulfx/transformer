//! Compose resource bounds before training-only search and six joint CE fits.
use crate::{
    adapter::Input,
    model::{Prior, Trained},
    *,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub search: search::Policy,
    pub max_programs: usize,
    pub radius_per_program: f64,
    pub minimum_mass: f64,
    /// Initial total mass here is per program; endpoints store the scaled policy.
    pub policies: Vec<joint::training::Policy>,
}
pub fn settings() -> Result<Settings> {
    Ok(serde_json::from_value(
        serde_json::from_str::<serde_json::Value>(PROTOCOL)?["fit"].clone(),
    )?)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Forecast {
    pub scalar_work: u64,
    pub buffer_bytes: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub prior: Prior,
    pub policy: joint::training::Policy,
    pub trained: Option<Trained>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub input: Input,
    pub settings: Settings,
    pub forecast: Forecast,
    pub search: Option<search::Search>,
    pub discovery: Option<discovery::Discovery>,
    pub endpoints: Vec<Endpoint>,
    pub error: Option<String>,
}

pub fn forecast(input: &Input, s: &Settings) -> Result<Forecast> {
    observations::validate(&input.observations, input.vocabulary)?;
    if !(1..=64).contains(&s.max_programs)
        || s.search.retained > s.max_programs
        || s.search.depth > 3
        || s.search.beam > 128
        || s.search.retained > 64
        || s.search.max_candidates > 10_000_000
        || !s.radius_per_program.is_finite()
        || !(1. ..=1e9 * 2.).contains(&s.radius_per_program)
        || !s.minimum_mass.is_finite()
        || !(1e-6..=1.).contains(&s.minimum_mass)
        || s.policies.len() != 3
        || s.policies.iter().map(|p| p.mode).collect::<Vec<_>>()
            != vec![
                joint::training::Mode::Forward,
                joint::training::Mode::Full,
                joint::training::Mode::Mini,
            ]
    {
        return Err("complete fit settings refused".into());
    }
    let n = input.observations.len() as u128;
    let lengths = input
        .observations
        .iter()
        .map(|r| (r.tokens.len() as u128 + 1).pow(2))
        .sum::<u128>();
    let bank = 302 + s.search.retained.max(6) as u128;
    let mut work = bank * (lengths * 4096 + n * 4096)
        + s.search.max_candidates as u128 * n * 128
        + s.search.depth as u128 * s.search.beam as u128 * lengths * 4096;
    let mut bytes = 1024u128.pow(3)
        + 64 * (n * 128 + bank * (2048 + n * 64))
        + 64 * (s.search.beam + s.search.retained) as u128 * (n * 32 + 16384)
        + 64 * input
            .observations
            .iter()
            .map(|r| r.tokens.len() as u128 * 8 + 128)
            .sum::<u128>();
    for base in &s.policies {
        let mut p = base.clone();
        p.initial_total_mass *= s.max_programs as f64;
        let f = p.forecast(n as usize, s.max_programs, input.vocabulary)?;
        work += 2 * f.scalar_work as u128;
        bytes += 2 * f.buffer_bytes as u128;
    }
    if 3 * work > 600_000_000_000_000 || bytes > joint::MAX_BYTES {
        return Err("complete search/fit/reconstruction forecast refused".into());
    }
    Ok(Forecast {
        scalar_work: work as u64,
        buffer_bytes: bytes as u64,
    })
}

pub fn construct(input: Input, settings: Settings) -> Result<Stage> {
    let forecast = forecast(&input, &settings)?;
    let mut error = None;
    let search = match search::run(
        &input.observations,
        input.vocabulary,
        settings.search.clone(),
    ) {
        Ok(s) => Some(s),
        Err(e) => {
            error = Some(e.to_string());
            None
        }
    };
    let discovery = if error.is_none() {
        let mut bank = program::numeric_bank();
        bank.extend(search.as_ref().unwrap().programs.clone());
        match discovery::run(
            &input.observations,
            observations::coordinates(input.vocabulary)?,
            bank,
        ) {
            Ok(d) => {
                if d.compatible.len() > settings.max_programs {
                    error = Some("retained program cap exceeded".into());
                }
                Some(d)
            }
            Err(e) => {
                error = Some(e.to_string());
                None
            }
        }
    } else {
        None
    };
    let mut endpoints = vec![];
    for prior in [Prior::Coordinates, Prior::Categorical] {
        for base in &settings.policies {
            let mut policy = base.clone();
            let p = discovery.as_ref().map_or(0, |d| d.compatible.len());
            policy.initial_total_mass *= p.max(1) as f64;
            let result = if let Some(e) = &error {
                Err(e.clone().into())
            } else {
                model::train(
                    discovery.as_ref().ok_or("missing discovery")?,
                    &input.observations,
                    prior,
                    policy.clone(),
                    settings.radius_per_program * p.max(1) as f64,
                    settings.minimum_mass,
                )
            };
            let (trained, error) = match result {
                Ok(t) => (Some(t), None),
                Err(e) => (None, Some(e.to_string())),
            };
            endpoints.push(Endpoint {
                prior,
                policy,
                trained,
                error,
            });
        }
    }
    Ok(Stage {
        input,
        settings,
        forecast,
        search,
        discovery,
        endpoints,
        error,
    })
}
