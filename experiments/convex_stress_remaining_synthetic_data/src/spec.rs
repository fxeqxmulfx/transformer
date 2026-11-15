//! Original generator identities, which are also the exact split-seed inputs.
use crate::*;
use serde_json::Value;

pub fn number(s: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        s[key]
            .as_u64()
            .ok_or("missing unsigned specification field")?,
    )
    .map_err(Into::into)
}
pub fn task(s: &Value) -> Result<&str> {
    s["task"]
        .as_str()
        .ok_or_else(|| "missing task family".into())
}
pub fn minimum(s: &Value) -> Result<usize> {
    number(
        s,
        if s["min_length"].is_null() {
            "length"
        } else {
            "min_length"
        },
    )
}
pub fn uses_numbers(s: &Value) -> bool {
    s["task"] == "count"
        || s["task"] == "random_lm"
        || (s["task"] == "addition" && s["index_hints"] == true)
}
pub fn symbols(s: &Value) -> Result<usize> {
    // Source Addition.controls omits the unused reserved symbol range without hints.
    if s["task"] == "addition" && s["index_hints"] == false {
        Ok(64)
    } else {
        number(s, "symbols")
    }
}
pub fn vocabulary(s: &Value) -> Result<usize> {
    if task(s)? == "crasp" {
        return Ok(36);
    }
    Ok(36
        + symbols(s)?
        + if uses_numbers(s) {
            number(s, "number_limit")? + 1
        } else {
            0
        })
}
pub fn validate(s: &Value, count: usize) -> Result<()> {
    let family = task(s)?;
    let length = number(s, "length")?;
    let low = minimum(s)?;
    if !["count", "addition", "crasp", "random_lm"].contains(&family)
        || low == 0
        || low > length
        || length > 4096
        || count == 0
        || count > 20000
        || count * (6 * length + 16) > 2_000_000
    {
        return Err("synthetic dimensions or total position budget refused".into());
    }
    if family == "crasp" {
        if low < 2
            || !(1..=5).contains(&number(s, "formula_depth")?)
            || number(s, "formula_seed")? > u32::MAX as usize
        {
            return Err("formula depth, seed or prefix length refused".into());
        }
    } else {
        let symbols = symbols(s)?;
        if !(2..=1024).contains(&symbols) {
            return Err("symbol alphabet refused".into());
        }
        if uses_numbers(s) {
            let limit = number(s, "number_limit")?;
            if !(1..=4096).contains(&limit)
                || length > limit
                || (family == "addition" && length + 1 > limit)
            {
                return Err("number range refused".into());
            }
        }
        if family == "addition"
            && (length > 32
                || !["forward", "reverse"].contains(&s["addition_order"].as_str().unwrap_or(""))
                || !["standard", "balanced"].contains(&s["carry_sampling"].as_str().unwrap_or(""))
                || !s["index_hints"].is_boolean()
                || (!s["carry_length"].is_null() && number(s, "carry_length")? > low))
        {
            return Err("addition format or exact integer-oracle bound refused".into());
        }
    }
    Ok(())
}
