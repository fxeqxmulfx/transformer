//! Source generation.py's final-token diagnostic stays separate from exact
//! accuracy: a correct final Mode answer need not imply a correct scratchpad.
use crate::*;
pub use generative::scoring::{Score, rollout, score};

pub fn final_answer(answer: &[i64], generated: &[usize], final_token: bool) -> Result<bool> {
    let ordinary = score(answer, generated)?;
    if !final_token {
        return Ok(ordinary.exact);
    }
    if answer.len() < 2 {
        return Err("final-token reference needs content before EOS".into());
    }
    Ok(ordinary.ended
        && generated.len() >= 2
        && generated[generated.len() - 2] == answer[answer.len() - 2] as usize)
}
