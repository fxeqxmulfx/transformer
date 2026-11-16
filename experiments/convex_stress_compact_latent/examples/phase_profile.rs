//! Fixed synthetic phase timings for task364; no original data or accuracy credit.
//! Uses the current complete model, preserving all 130 rows and the 127+3 tail.
use convex_stress_compact_latent::{
    Parameters, Record, Result, Shape, Work,
    domain::Domain,
    head::{Head, PredictionBudget},
    persistence, projection,
    training::{self, Config, Mode},
};
use sha2::{Digest, Sha256};
use std::{hint::black_box, io::Write, path::PathBuf, process::Command, time::Instant};

const LIMIT: u64 = 128 * 1024 * 1024;
const PROCESS: u64 = 4 * 1024u64.pow(3);
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
fn resources(guard: &legacy::resources::Guard) -> Result<()> {
    guard.check()?;
    let out = Command::new("ps").args(["-e", "-o", "rss="]).output()?;
    if !out.status.success() {
        return Err("cannot read aggregate RSS".into());
    }
    let mut rss = 0u64;
    for value in std::str::from_utf8(&out.stdout)?.split_whitespace() {
        rss = rss
            .checked_add(
                value
                    .parse::<u64>()?
                    .checked_mul(1024)
                    .ok_or("RSS overflow")?,
            )
            .ok_or("aggregate RSS overflow")?;
    }
    if rss + PROCESS > 110_000_000_000 {
        return Err("aggregate RAM cannot admit the bounded phase profile".into());
    }
    Ok(())
}
fn sample<T>(
    guard: &legacy::resources::Guard,
    mut f: impl FnMut() -> Result<T>,
) -> Result<(T, u64)> {
    let (mut values, mut final_value) = (Vec::new(), None);
    for _ in 0..3 {
        drop(final_value.take());
        resources(guard)?;
        let start = Instant::now();
        let value = f()?;
        values.push(start.elapsed().as_nanos() as u64);
        final_value = Some(black_box(value));
    }
    values.sort_unstable();
    Ok((final_value.unwrap(), values[1]))
}
struct Hasher(Sha256);
impl Write for Hasher {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn endpoint(d: &Domain, trained: &training::Trained) -> Result<String> {
    let mut bytes = Hasher(Sha256::new());
    persistence::write(d, &trained.parameters, &mut bytes, &mut work())?;
    // Bind every logical coefficient, CE and step word, independent of storage.
    for p in 0..d.shape.programs {
        for f in 0..d.shape.states {
            for c in 0..d.shape.classes {
                let (re, im) = trained.head.coefficient(p, f, c)?;
                bytes.write_all(&re.to_bits().to_le_bytes())?;
                bytes.write_all(&im.to_bits().to_le_bytes())?;
            }
        }
    }
    for x in [trained.initial_ce, trained.final_ce] {
        bytes.write_all(&x.to_bits().to_le_bytes())?;
    }
    for step in &trained.steps {
        for x in [
            step.before_ce,
            step.after_ce,
            step.before_mass,
            step.after_mass,
            step.parameter_change_l2,
            step.maximum_feasibility_residual,
            step.maximum_kkt_residual,
        ] {
            bytes.write_all(&x.to_bits().to_le_bytes())?;
        }
    }
    Ok(hex(&bytes.0.finalize()))
}
fn config(mode: Mode) -> Config {
    Config {
        mode,
        epochs: if mode == Mode::Mini { 1 } else { 0 },
        batch_size: 127,
        learning_rate: 3.,
        initial_mass: 8.,
    }
}
fn main() -> Result<()> {
    let workspace = PathBuf::from(
        std::env::var_os("SHARED_COLUMNS_WORKSPACE").ok_or("set SHARED_COLUMNS_WORKSPACE")?,
    );
    let storage = PathBuf::from(
        std::env::var_os("SHARED_COLUMNS_STORAGE").ok_or("set SHARED_COLUMNS_STORAGE")?,
    );
    let guard = legacy::resources::Guard::new(&workspace, &storage, 1024 * 1024)?;
    resources(&guard)?;
    for programs in [128, 512] {
        for classes in [97, 193] {
            resources(&guard)?;
            let shape = Shape {
                programs,
                states: 8,
                classes,
            };
            let rows = (0..130)
                .map(|i| Record {
                    label: 0,
                    states: (0..programs).map(|p| 4 * ((i + p) % 2)).collect(),
                })
                .collect::<Vec<_>>();
            let d = Domain::new(shape, &rows, 1., 32., &mut work())?;
            let q = Parameters::initial(&d, 8., LIMIT, &mut work())?;
            let (head, head_ns) = sample(&guard, || Head::build(&d, &q, LIMIT, &mut work()))?;
            let (_, loss_ns) = sample(&guard, || {
                training::loss_minibatches(
                    &head,
                    convex_stress_compact_latent::records::Source::Expanded(&rows),
                    127,
                    &mut work(),
                )
            })?;
            let ((_, gradient), gradient_ns) = sample(&guard, || {
                training::gradient(&d, &head, &rows[..127], LIMIT, &mut work())
            })?;
            let (proposal, proposal_ns) = sample(&guard, || {
                projection::Target::step(&d, &q, &gradient, 3., LIMIT, &mut work())
            })?;
            let (_, projection_ns) = sample(&guard, || {
                projection::project(&d, &proposal, LIMIT, &mut work())
            })?;
            let known = vec![0; programs];
            let unknown = vec![1; programs];
            let (known_prediction, known_ns) = sample(&guard, || {
                head.predict(&known, &mut PredictionBudget::new(1)?, &mut work())
            })?;
            let (unknown_prediction, unknown_ns) = sample(&guard, || {
                head.predict(&unknown, &mut PredictionBudget::new(1)?, &mut work())
            })?;
            let (forward, forward_ns) = sample(&guard, || {
                training::train(&d, &rows, config(Mode::Forward), LIMIT, &mut work())
            })?;
            let forward_hash = endpoint(&d, &forward)?;
            let forward_work = forward.operations;
            drop(forward);
            let (mini, mini_ns) = sample(&guard, || {
                training::train(&d, &rows, config(Mode::Mini), LIMIT, &mut work())
            })?;
            assert_eq!(mini.steps.len(), 2);
            assert_eq!(
                (
                    mini.steps[1].first_record,
                    mini.steps[1].end_record_exclusive
                ),
                (127, 130)
            );
            let mini_hash = endpoint(&d, &mini)?;
            println!(
                concat!(
                    "{{\"programs\":{},\"states\":8,\"classes\":{},\"records\":130,",
                    "\"rounds\":3,\"head_build_ns\":{},\"loss_130_ns\":{},\"gradient_127_ns\":{},",
                    "\"proposal_ns\":{},\"projection_ns\":{},\"known_predict_ns\":{},",
                    "\"unknown_predict_ns\":{},\"known_certificate\":\"{:?}\",",
                    "\"unknown_certificate\":\"{:?}\",\"forward_total_ns\":{},\"mini_total_ns\":{},",
                    "\"forward_work\":{},\"mini_work\":{},\"forward_sha256\":\"{}\",",
                    "\"mini_sha256\":\"{}\",\"original_data_used\":false}}"
                ),
                programs,
                classes,
                head_ns,
                loss_ns,
                gradient_ns,
                proposal_ns,
                projection_ns,
                known_ns,
                unknown_ns,
                known_prediction.certificate,
                unknown_prediction.certificate,
                forward_ns,
                mini_ns,
                forward_work,
                mini.operations,
                forward_hash,
                mini_hash
            );
        }
    }
    resources(&guard)
}
