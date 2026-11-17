use baseline::channel::{argmax, cross_entropy};
use composed_retrieval::{Result, model::Model, operator::Tables, radial};
use serde_json::{Value, json};

const VALUES: [f64; 5] = [1., 0.75, 0.5, 0.25, 0.];
pub struct Row {
    pub pair: [usize; 2],
    pub target: usize,
}

#[derive(Default, Clone, Copy)]
struct Counts {
    loss: f64,
    correct: usize,
    lower: Option<f64>,
    upper: Option<f64>,
    other_violations: usize,
}
impl Counts {
    fn row(z: &[f64], y: usize, k: usize) -> Self {
        let mut out = Self {
            loss: cross_entropy(z, y),
            correct: usize::from(argmax(z) == y),
            ..Self::default()
        };
        if y == k {
            let best = z
                .iter()
                .enumerate()
                .filter(|&(c, _)| c != k)
                .map(|(_, &s)| s)
                .fold(f64::NEG_INFINITY, f64::max);
            out.lower = Some(best - z[k]);
        } else {
            out.upper = Some(z[y] - z[k]);
            out.other_violations = usize::from(
                z.iter()
                    .enumerate()
                    .any(|(c, &s)| c != k && c != y && s >= z[y]),
            );
        }
        out
    }
    fn add(&mut self, row: Self) {
        self.loss += row.loss;
        self.correct += row.correct;
        self.other_violations += row.other_violations;
        if let Some(v) = row.lower {
            self.lower = Some(self.lower.map_or(v, |l| l.max(v)));
        }
        if let Some(v) = row.upper {
            self.upper = Some(self.upper.map_or(v, |u| u.min(v)));
        }
    }
    fn report(self, n: usize, offset: f64, bound: f64, radius: f64) -> Value {
        let feasible = self.other_violations == 0
            && self.lower.zip(self.upper).is_none_or(|(l, u)| l < u)
            && self.lower.is_none_or(|l| l < bound - offset)
            && self.upper.is_none_or(|u| -bound - offset < u);
        json!({"radius":radius,"correct":self.correct,"ce":self.loss/n as f64,"strict_lower":self.lower,
            "strict_upper":self.upper,"other_rival_violations":self.other_violations,"offset_box_feasible":feasible})
    }
}

/// Score only the original training rows. The two focal object IDs are
/// selected independently by training-error frequency, then by smaller ID.
pub fn measure(m: &Model, rows: &[Row], raw: &[u8], baseline: &Value) -> Result<Value> {
    let v = m.vocabulary;
    if rows.is_empty() || v < 2 || raw.len() != rows.len() * v * 8 {
        return Err("invalid training array dimensions".into());
    }
    let table = Tables::new(m, &[1])?;
    let mut misses = [vec![0usize; v], vec![0usize; v]];
    let (mut correct, mut loss) = (0usize, 0.);
    let vectors = || {
        raw.chunks_exact(v * 8).map(|b| {
            b.as_chunks::<8>()
                .0
                .iter()
                .map(|&x| f64::from_le_bytes(x))
                .collect::<Vec<_>>()
        })
    };
    for (r, z) in rows.iter().zip(vectors()) {
        if r.target >= v || r.pair.iter().any(|&id| id >= v) {
            return Err("invalid source object".into());
        }
        let physical = table.logits(m, &table.pass(m, &r.pair).hidden);
        if z.iter()
            .zip(&physical)
            .any(|(&a, &b)| a.to_bits() != b.to_bits() || !a.is_finite())
        {
            return Err("training logits differ from the sealed physical model".into());
        }
        loss += cross_entropy(&z, r.target);
        if argmax(&z) == r.target {
            correct += 1;
        } else {
            let [a, q] = r.pair;
            misses[0][a] += 1;
            if a != q {
                misses[0][q] += 1;
            }
            misses[1][r.target] += 1;
        }
    }
    let ce = loss / rows.len() as f64;
    if baseline["targets"] != rows.len()
        || baseline["correct"] != correct
        || (baseline["ce"].as_f64().ok_or("missing original train CE")? - ce).abs() > 1e-12
    {
        return Err("original training endpoint counts or CE differ".into());
    }
    if correct == rows.len() {
        return Ok(json!({"targets":rows.len(),"correct":correct,"ce":ce,"interventions":[]}));
    }
    let [operand, class] = misses.each_ref().map(|counts| {
        (0..v)
            .max_by_key(|&k| (counts[k], std::cmp::Reverse(k)))
            .expect("nonempty original vocabulary")
    });
    let mut counts = [Counts::default(); VALUES.len()];
    let mut radii = vec![1.; v];
    let mut affected = 0;
    for (r, z) in rows.iter().zip(vectors()) {
        let base = Counts::row(&z, r.target, class);
        if !r.pair.contains(&operand) {
            for c in &mut counts {
                c.add(base);
            }
            continue;
        }
        affected += 1;
        let pass = table.pass(m, &r.pair);
        for (&value, c) in VALUES.iter().zip(&mut counts) {
            radii[operand] = value;
            let adjusted = radial::hidden(m, &table, &pass, &r.pair, &radii)?;
            let radial_logits = table.logits(m, &adjusted);
            if value == 1. && radial_logits != z {
                return Err("unit intervention changed logits".into());
            }
            c.add(Counts::row(&radial_logits, r.target, class));
        }
    }
    let bound = m.output_offset_bound.ok_or("missing offset domain")?;
    let offset = m.weights[table.layout.output_offsets.start + class];
    let interventions: Vec<_> = counts
        .into_iter()
        .zip(VALUES)
        .map(|(c, r)| c.report(rows.len(), offset, bound, r))
        .collect();
    Ok(
        json!({"targets":rows.len(),"correct":correct,"ce":ce,"focal_operand":operand,"focal_class":class,
        "operand_error_rows":misses[0][operand],"class_error_rows":misses[1][class],"affected_rows":affected,
        "focal_offset":offset,"offset_domain":[-bound,bound],"interventions":interventions}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_interval_uses_remaining_offset_room_and_each_unmoved_rival() {
        let mut c = Counts::row(&[2., 3., 1.], 0, 0);
        c.add(Counts::row(&[1., 5., 3.], 1, 0));
        assert_eq!(c.report(2, 0., 2., 0.)["offset_box_feasible"], true);
        assert_eq!(c.report(2, 1., 2., 0.)["offset_box_feasible"], false);
        c.add(Counts::row(&[1., 3., 3.], 1, 0));
        assert_eq!(c.report(3, 0., 2., 0.)["offset_box_feasible"], false);
    }
}
