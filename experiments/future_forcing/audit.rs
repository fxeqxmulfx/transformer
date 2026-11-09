//! Separate phase: inspect later traces only after every forecast is frozen.
use crate::{
    ANCHORS, Result, SEEDS,
    classes::Partition,
    evidence::{hash, require},
    forecast::{self, Anchor, E, NAMES, RELATIVE, THRESHOLDS, norm},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};
fn read(p: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn number(v: &Value) -> Result<f64> {
    let x = v.as_f64().ok_or("missing numeric field")?;
    require(x.is_finite(), "nonfinite bound")?;
    Ok(x)
}
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * (1. + a.abs().max(b.abs()))
}
fn digest(h: Sha256) -> String {
    h.finalize().iter().map(|v| format!("{v:02x}")).collect()
}
fn row(r: &mut BufReader<File>, h: &mut Sha256) -> Result<Vec<f64>> {
    let mut b = vec![0; 4 * E];
    r.read_exact(&mut b)?;
    h.update(&b);
    let v: Vec<_> = b
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b) as f64)
        .collect();
    require(v.iter().all(|v| v.is_finite()), "nonfinite embedding")?;
    Ok(v)
}
struct Window {
    anchor: usize,
    end: usize,
    initial: Anchor,
    projected: Vec<f64>,
    reader: BufReader<File>,
    path: PathBuf,
    hash: String,
    previous_q: f64,
    previous: [[f64; 8]; 2],
    prefixes: usize,
    longest: [[usize; 4]; 2],
    above: [[usize; 4]; 2],
    minimum: [f64; 2],
    observed_min: f64,
    max_excess: [f64; 5],
    violations: [usize; 5],
    moving: usize,
}
impl Window {
    fn new(root: &Path, path: &Path, case: &Value, active: &[bool]) -> Result<Self> {
        let meta = read(&path.join("frozen.json"))?;
        require(
            meta == case["construction"] && meta["frozen_before_trace_reads"] == true,
            "frozen metadata changed",
        )?;
        let anchor = case["anchor"].as_u64().ok_or("missing anchor")? as usize;
        let end = (anchor + 4000).min(10000);
        require(
            meta["anchor"] == anchor && meta["end"] == end,
            "window cadence changed",
        )?;
        let state = fs::read(root.join(format!("step-{anchor:06}/state.bin")))?;
        require(hash(&state) == meta["state_sha256"], "anchor changed")?;
        let initial = Anchor::new(forecast::state(&state, anchor)?, active.to_vec())?;
        require(
            initial.metadata() == meta["initial"],
            "initial cone or geometry changed",
        )?;
        let projected = Partition::new(active.to_vec(), 128).project(&initial.state.p)[0].clone();
        let path = path.join("forecast.jsonl");
        let expected = meta["forecast_sha256"]
            .as_str()
            .ok_or("missing forecast hash")?
            .to_string();
        require(
            hash(&fs::read(&path)?) == expected,
            "forecast changed before audit",
        )?;
        let mut previous = [[0.; 8]; 2];
        for t in &mut previous {
            t[4] = initial.state.p.iter().map(|x| x.abs()).fold(0., f64::max);
        }
        Ok(Self {
            anchor,
            end,
            initial,
            projected,
            reader: BufReader::new(File::open(&path)?),
            path,
            hash: expected,
            previous_q: 1.,
            previous,
            prefixes: 0,
            longest: [[0; 4]; 2],
            above: [[0; 4]; 2],
            minimum: [1.; 2],
            observed_min: 1.,
            max_excess: [0.; 5],
            violations: [0; 5],
            moving: 0,
        })
    }
    fn check(&mut self, kind: usize, actual: f64, bound: f64) {
        let excess = (actual - bound) / (1. + actual.abs().max(bound.abs()));
        self.max_excess[kind] = self.max_excess[kind].max(excess);
        self.violations[kind] += usize::from(excess > 1e-8);
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Auditor passes independently computed current update and class modes"
    )]
    fn observe(
        &mut self,
        step: usize,
        eta: f64,
        new: &[f64],
        projected: &[Vec<f64>; 3],
        forcing: [f64; 2],
        rounding: f64,
        moving: bool,
    ) -> Result<()> {
        if step <= self.anchor || step > self.end {
            return Ok(());
        }
        let h = step - self.anchor;
        let mut line = String::new();
        require(
            self.reader.read_line(&mut line)? > 0,
            "missing frozen prefix",
        )?;
        let frozen: Value = serde_json::from_str(&line)?;
        require(
            frozen[0] == step && number(&frozen[1])? == eta,
            "forecast step/rate changed",
        )?;
        let q = number(&frozen[2])?;
        let decay = 1. - eta * 0.1;
        require(close(q, self.previous_q * decay), "decay recurrence failed")?;
        let distance = norm(
            &new.iter()
                .zip(&self.initial.state.p)
                .map(|(x, a)| x - q * a)
                .collect::<Vec<_>>(),
        );
        let active_distance = norm(
            &projected[0]
                .iter()
                .zip(&self.projected)
                .map(|(x, a)| x - q * a)
                .collect::<Vec<_>>(),
        );
        let an = norm(&projected[0]);
        let cn = norm(&projected[1]).hypot(norm(&projected[2]));
        let actual = an * an / (an * an + cn * cn);
        self.observed_min = self.observed_min.min(actual);
        self.moving += usize::from(moving);
        let maximum = new.iter().map(|x| x.abs()).fold(0., f64::max);
        for m in 0..2 {
            let raw = frozen[3][m].as_array().ok_or("missing track")?;
            require(raw.len() == 8, "track shape changed")?;
            let mut t = [0.; 8];
            for (x, v) in t.iter_mut().zip(raw) {
                *x = number(v)?;
                require(*x >= 0., "negative forecast quantity")?;
            }
            let before = self.previous[m];
            let memory = (81f64 / 95.).powi(h as i32);
            let bias = (1. - 0.95f64.powi(step as i32)).sqrt() / (1. - 0.9f64.powi(step as i32));
            for (i, sum, dim) in [
                (0, self.initial.sum_ratio[0], E as f64),
                (1, self.initial.sum_ratio[1], 4224.),
                (2, self.initial.max_ratio, 1.),
            ] {
                let sum = if m == 0 {
                    dim * 19. / 14. + (sum - dim * 19. / 14.) * memory
                } else {
                    dim * 19. / 14.
                };
                require(
                    close(t[i].powi(2), sum * bias * bias),
                    "future direction formula changed",
                )?;
            }
            let predicted_round =
                8. * f32::EPSILON as f64 * (1. + before[4] + eta * t[2] * (1. + RELATIVE));
            require(
                close(t[3], predicted_round)
                    && close(
                        t[4],
                        decay * before[4] + eta * t[2] * (1. + RELATIVE) + t[3],
                    ),
                "coordinate cap recurrence failed",
            )?;
            require(
                close(
                    t[5],
                    decay * before[5] + eta * t[1] * (1. + RELATIVE) + 4224f64.sqrt() * t[3],
                ) && close(
                    t[6],
                    decay * before[6] + eta * t[0] * (1. + RELATIVE) + (E as f64).sqrt() * t[3],
                ),
                "tube recurrence failed",
            )?;
            let lower = (q * self.initial.initial[0] - t[5]).max(0.);
            let upper = q * self.initial.initial[1] + t[6];
            let predicted = if lower == 0. {
                0.
            } else {
                lower * lower / (lower * lower + upper * upper)
            };
            require(close(t[7], predicted), "share reduction changed")?;
            self.check(
                0,
                forcing[0],
                eta * t[0] * (1. + RELATIVE) + (E as f64).sqrt() * t[3],
            );
            self.check(
                0,
                forcing[1],
                eta * t[1] * (1. + RELATIVE) + 4224f64.sqrt() * t[3],
            );
            self.check(1, rounding, t[3]);
            self.check(2, maximum, t[4]);
            self.check(3, distance, t[6]);
            self.check(3, active_distance, t[5]);
            self.check(4, t[7], actual);
            self.minimum[m] = self.minimum[m].min(t[7]);
            for (j, threshold) in THRESHOLDS.into_iter().enumerate() {
                if t[7] > threshold + 1e-8 {
                    self.above[m][j] += 1;
                    if self.longest[m][j] == h - 1 {
                        self.longest[m][j] = h;
                    }
                }
            }
            self.previous[m] = t;
        }
        self.previous_q = q;
        self.prefixes += 1;
        Ok(())
    }
    fn finish(mut self) -> Result<Value> {
        require(
            self.prefixes == self.end - self.anchor
                && self.reader.read_line(&mut String::new())? == 0,
            "incomplete prefix inventory",
        )?;
        require(
            hash(&fs::read(&self.path)?) == self.hash,
            "forecast changed during audit",
        )?;
        let tracks:Vec<_>=(0..2).map(|m|json!({"method":NAMES[m],"longest_prefix":self.longest[m],"prefixes_above":self.above[m],
            "minimum_share_lower":self.minimum[m],"endpoint_share_lower":self.previous[m][7],"endpoint_radii":[self.previous[m][5],self.previous[m][6]]})).collect();
        Ok(
            json!({"status":if self.violations.iter().sum::<usize>()==0 {"pass"} else {"fail"},"anchor":self.anchor,"end":self.end,
            "prefixes":self.prefixes,"moving_updates":self.moving,"minimum_observed_share":self.observed_min,"tracks":tracks,
            "check_kinds":["forcing","rounding","parameter","tube","share"],"max_scaled_excess":self.max_excess,"violations":self.violations}),
        )
    }
}
fn seed(
    root: &Path,
    out: &Path,
    cases: &[Value],
    active: &[bool],
    reference: &Value,
    cohort: &str,
    id: u64,
) -> Result<Vec<Value>> {
    let mut windows = vec![];
    for (i, anchor) in ANCHORS.into_iter().enumerate() {
        let c = &cases[i];
        require(
            c["cohort"] == cohort
                && c["seed"] == id
                && c["anchor"] == anchor
                && c["status"] == "frozen",
            "declared window changed",
        )?;
        windows.push(Window::new(
            root,
            &out.join(format!("{cohort}-seed{id}-anchor{anchor}")),
            c,
            active,
        )?);
    }
    let part = Partition::new(active.to_vec(), 128);
    let mut trace = BufReader::new(File::open(root.join("embeddings.bin"))?);
    let mut th = Sha256::new();
    let mut header = [0; 24];
    trace.read_exact(&mut header)?;
    th.update(header);
    require(
        header
            .as_chunks::<8>()
            .0
            .iter()
            .map(|x| u64::from_le_bytes(*x))
            .collect::<Vec<_>>()
            == [1, 10001, E as u64],
        "wrong trace shape",
    )?;
    let mut old = row(&mut trace, &mut th)?;
    require(
        windows[0].initial.state.p == old,
        "initial trace/checkpoint mismatch",
    )?;
    let mut updates = BufReader::new(File::open(root.join("updates.jsonl"))?);
    let mut uh = Sha256::new();
    for step in 1..=10000 {
        let mut line = String::new();
        require(updates.read_line(&mut line)? > 0, "missing update row")?;
        uh.update(line.as_bytes());
        let r: Value = serde_json::from_str(&line)?;
        let eta = number(&r["eta"])?;
        require(
            r["step"] == step && eta == forecast::rate(cohort, step - 1),
            "source step/rate mismatch",
        )?;
        let new = row(&mut trace, &mut th)?;
        for window in &windows {
            if step == window.anchor {
                require(
                    window.initial.state.p == new,
                    "anchor trace/checkpoint mismatch",
                )?;
            }
        }
        let decay = 1. - eta * 0.1;
        let f: Vec<_> = new.iter().zip(&old).map(|(n, o)| n - decay * o).collect();
        let forcing = [norm(&f), norm(&part.project(&f)[0])];
        let p = part.project(&new);
        let mut moving = false;
        for k in 0..E {
            moving |= new[k] != old[k];
        }
        for (j, mode) in p.iter().enumerate() {
            require(
                close(norm(mode).powi(2), number(&r["after"][j])?),
                "independent energy mismatch",
            )?;
        }
        let rounding = number(&r["rounding_max"])?;
        for w in &mut windows {
            w.observe(step, eta, &new, &p, forcing, rounding, moving)?;
        }
        old = new;
    }
    require(
        trace.read(&mut [0])? == 0 && updates.read_line(&mut String::new())? == 0,
        "extra source rows",
    )?;
    require(
        reference["trace_sha256"] == digest(th) && reference["updates_sha256"] == digest(uh),
        "source hashes changed",
    )?;
    windows.into_iter().map(Window::finish).collect()
}
pub fn all(out: &Path) -> Result<()> {
    let started = Instant::now();
    let report = read(&out.join("report.json"))?;
    require(
        report["status"] == "frozen"
            && report["declared_cases"] == 144
            && report["declared_prefixes"] == 444000
            && report["cases"].as_array().is_some_and(|v| v.len() == 144),
        "incomplete construction",
    )?;
    for (path, h) in report["inputs"]["files"]
        .as_object()
        .ok_or("missing ledger")?
    {
        require(
            hash(&fs::read(path)?) == h.as_str().unwrap(),
            "constructor input changed",
        )?;
    }
    let cases = report["cases"].as_array().unwrap();
    let mut results = vec![];
    let mut slot = 0;
    for (ci, cohort) in ["lr001", "lr01", "settle"].into_iter().enumerate() {
        require(
            report["cohorts"][ci]["name"] == cohort,
            "cohort order changed",
        )?;
        let root = Path::new(
            report["cohorts"][ci]["path"]
                .as_str()
                .ok_or("missing cohort path")?,
        );
        let source = read(&root.join("report.json"))?;
        for (si, id) in SEEDS.into_iter().enumerate() {
            let sr = root.join(format!("seed{id}"));
            let active = crate::support(&read(&sr.join("step-000000/queries.json"))?)?;
            let outcome = seed(
                &sr,
                out,
                &cases[slot..slot + 6],
                &active,
                &source["seeds"][si],
                cohort,
                id,
            );
            match outcome {
                Ok(rows) => {
                    for mut r in rows {
                        r["cohort"] = json!(cohort);
                        r["seed"] = json!(id);
                        results.push(r);
                    }
                }
                Err(e) => {
                    for anchor in ANCHORS {
                        results.push(json!({"cohort":cohort,"seed":id,"anchor":anchor,"status":"error","error":e.to_string()}));
                    }
                }
            }
            slot += 6;
            eprintln!(
                "verified future cohort={cohort} seed={id} elapsed={:.1}s",
                started.elapsed().as_secs_f64()
            );
        }
    }
    for (path, h) in report["inputs"]["files"].as_object().unwrap() {
        require(
            hash(&fs::read(path)?) == h.as_str().unwrap(),
            "constructor input changed during audit",
        )?;
    }
    let failed = results.iter().filter(|r| r["status"] != "pass").count();
    let combined = number(&report["construction_seconds"])? + started.elapsed().as_secs_f64();
    let prefixes: usize = results
        .iter()
        .map(|r| r["prefixes"].as_u64().unwrap_or(0) as usize)
        .sum();
    let passed = failed == 0 && prefixes == 444000 && combined <= 1800.;
    println!(
        "{}",
        json!({"passed":passed,"failed":failed,"cases":results,"prefixes":prefixes,"seconds":started.elapsed().as_secs_f64(),
        "combined_postprocessing_seconds":combined,"source_sha256":report["source_sha256"],"constructor_inputs_verified":true})
    );
    require(passed, "future audit failed; every window retained")
}
