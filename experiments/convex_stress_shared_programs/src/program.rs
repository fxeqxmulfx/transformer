//! CM52's fixed input-only bank. All marker candidates are vocabulary symbols;
//! no token has a designated task role and no output label enters execution.
use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Program {
    Lag(usize),
    Walk {
        hops: usize,
        offset: usize,
        marker: Option<usize>,
        latest: bool,
    },
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Value {
    pub state: usize,
    pub missing: bool,
    pub positions: u64,
}

pub fn catalogue(vocabulary: usize, context: usize) -> Result<Vec<Program>> {
    if !(2..=1024).contains(&vocabulary)
        || !(1..=4096).contains(&context)
        || context + 32 * (vocabulary + 1) > 16_384
    {
        return Err("bounded input program catalogue refused".into());
    }
    let mut out: Vec<_> = (0..context).map(Program::Lag).collect();
    for hops in 1..=4 {
        for offset in 1..=4 {
            for marker in std::iter::once(None).chain((0..vocabulary).map(Some)) {
                for latest in [false, true] {
                    out.push(Program::Walk {
                        hops,
                        offset,
                        marker,
                        latest,
                    });
                }
            }
        }
    }
    Ok(out)
}

impl Program {
    pub fn run(self, tokens: &[usize]) -> Value {
        match self {
            Self::Lag(lag) => match tokens.len().checked_sub(lag + 1) {
                Some(i) => Value {
                    state: tokens[i],
                    missing: false,
                    positions: 1,
                },
                None => Value {
                    state: 0,
                    missing: true,
                    positions: 1,
                },
            },
            Self::Walk {
                hops,
                offset,
                marker,
                latest,
            } => {
                let mut value = Value {
                    state: *tokens.last().unwrap(),
                    ..Value::default()
                };
                let end = tokens.len() - 1;
                for _ in 0..hops {
                    let mut found = None;
                    for i in 0..end.saturating_sub(offset) {
                        value.positions += 1;
                        if tokens[i] == value.state
                            && marker.is_none_or(|m| i > 0 && tokens[i - 1] == m)
                        {
                            found = Some(tokens[i + offset]);
                            if !latest {
                                break;
                            }
                        }
                    }
                    match found {
                        Some(state) => value.state = state,
                        None => {
                            value.state = 0;
                            value.missing = true;
                            break;
                        }
                    }
                }
                value
            }
        }
    }

    /// Independent window/list interpreter, used for retained programs and
    /// rejected-program witnesses. It shares no control flow with run's scan.
    pub fn interpret(self, tokens: &[usize]) -> (usize, bool) {
        match self {
            Self::Lag(lag) => tokens
                .iter()
                .rev()
                .nth(lag)
                .map_or((0, true), |&s| (s, false)),
            Self::Walk {
                hops,
                offset,
                marker,
                latest,
            } => {
                let mut key = *tokens.last().unwrap();
                let earlier = &tokens[..tokens.len() - 1];
                for _ in 0..hops {
                    let matches: Vec<_> = earlier
                        .windows(offset + 1)
                        .enumerate()
                        .filter(|(i, w)| {
                            w[0] == key && marker.is_none_or(|m| *i > 0 && earlier[*i - 1] == m)
                        })
                        .map(|(_, w)| w[offset])
                        .collect();
                    let chosen = if latest {
                        matches.last()
                    } else {
                        matches.first()
                    };
                    match chosen {
                        Some(&s) => key = s,
                        None => return (0, true),
                    }
                }
                (key, false)
            }
        }
    }
}
