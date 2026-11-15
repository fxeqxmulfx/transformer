//! CM59 necessary count ranges relative to the retained opposite child bank.
//! These preserve possible partners for that bank, not globally missing anchors.
use crate::{bank::Bank, latent::Meter, pool::Pool, transitions::Config, *};
use std::collections::BTreeMap;
#[derive(Clone, Debug)]
pub struct Ranges {
    pub low: Vec<i16>,
    pub high: Vec<i16>,
}
impl Ranges {
    pub fn allows(&self, i: usize, value: i16) -> bool {
        self.low[i] <= value && value <= self.high[i]
    }
    pub fn impossible(&self) -> bool {
        self.low.iter().zip(&self.high).any(|(a, b)| a > b)
    }
}
pub fn partners(
    pool: &Pool,
    observed: &BTreeMap<usize, bool>,
    anchor: &Bank,
    config: &Config,
    extras: &[Vec<i16>],
    seek_right: bool,
    meter: &mut Meter,
) -> Result<Ranges> {
    if anchor.rows.is_empty() {
        return Err("count range requires a retained anchor".into());
    }
    meter.charge(
        anchor.rows.len() as u64 * pool.nodes.len() as u64 * 32 + pool.nodes.len() as u64 * 256,
    )?;
    let mut low = vec![i16::MAX; pool.nodes.len()];
    let mut high = vec![i16::MIN; pool.nodes.len()];
    for a in &anchor.rows {
        for ((lo, hi), &count) in low.iter_mut().zip(&mut high).zip(&a.counts) {
            *lo = (*lo).min(count);
            *hi = (*hi).max(count);
        }
    }
    let mut out = Ranges {
        low: vec![0; pool.nodes.len()],
        high: pool.nodes.iter().map(|n| n.depth as i16).collect(),
    };
    for (&i, &target) in observed {
        if !config.passes(pool.nodes[i].symbol) {
            continue;
        }
        let wanted = target ^ config.flipped ^ config.negated;
        let extra = config.extra.map_or(0, |s| i32::from(extras[s][i]));
        let (lo, hi, k) = (i32::from(low[i]), i32::from(high[i]), config.offset);
        let (l, u) = if seek_right {
            if wanted {
                (lo - extra - k + 1, i32::from(out.high[i]))
            } else {
                (0, hi - extra - k)
            }
        } else if wanted {
            (0, hi + extra + k - 1)
        } else {
            (lo + extra + k, i32::from(out.high[i]))
        };
        out.low[i] = i16::try_from(l.max(0))?;
        out.high[i] = i16::try_from(u.min(i32::from(out.high[i])))?;
    }
    Ok(out)
}
