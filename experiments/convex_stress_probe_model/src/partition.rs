//! CM64 retained_domain_eq: a conflicting program has exactly zero mass.
//! Keep every other catalog member, including duplicates and unknown states.
//! Each rejection has two original supervised IDs and different observed labels.
use crate::{
    Result, Work, features,
    input::{self, Observation, Prepared},
    reference,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conflict {
    pub state: usize,
    pub first: Observation,
    pub second: Observation,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Keep,
    Conflict(Conflict),
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Partition {
    pub decisions: Vec<Decision>,
    pub retained: Vec<usize>,
}
impl Partition {
    pub fn allocation_bytes(&self) -> u128 {
        input::bytes(&self.decisions) + input::bytes(&self.retained)
    }
}
pub fn build(input: &Prepared<'_>, work: &mut Work) -> Result<Partition> {
    let mut decisions = input::vector(input.programs.len())?;
    let mut retained = input::vector(input.programs.len())?;
    for (p, &program) in input.programs.iter().enumerate() {
        let mut observed = [None::<usize>; features::STATES];
        let mut decision = Decision::Keep;
        for (i, &record) in input.observations().iter().enumerate() {
            let prefix = input.prefix(i)?;
            let state = features::state(program, &input.alphabet, prefix, work)?;
            if state >= features::STATES
                || state != reference::state(program, &input.alphabet, prefix, work)?
            {
                return Err("independent causal state interpretations disagree".into());
            }
            if let Some(first) = observed[state] {
                if input.observations()[first].label != record.label && decision == Decision::Keep {
                    decision = Decision::Conflict(Conflict {
                        state,
                        first: input.observations()[first],
                        second: record,
                    });
                }
            } else {
                observed[state] = Some(i);
            }
        }
        if decision == Decision::Keep {
            retained.push(p);
        }
        decisions.push(decision);
    }
    Ok(Partition {
        decisions,
        retained,
    })
}
