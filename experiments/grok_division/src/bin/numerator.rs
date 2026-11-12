//! Reads the held-out predictions that runs recorded and reports them on the equations with
//! numerator zero, `0 / y = 0`, beside all held-out equations.
//!
//!     numerator [--list] RUN_DIR...
//!
//! For each run it prints one line of JSON:
//!
//! * `zero_numerator`: how many held-out equations have numerator zero;
//! * `zero_all_right_first`: the first recorded step at which every one of them is answered
//!   rightly, and `heldout_right_then`: the right held-out answers of that step;
//! * `zero_all_right_sustained`: the first step from which they are at every later recorded step;
//! * `zero_all_right_evaluations`: the evaluations that have them all right;
//! * `later_*`, of the evaluations after that first one: how many there are, how many leave a
//!   training equation wrong, how many have one of them wrong, how many of these leave a training
//!   equation wrong, the largest training accuracy among these, and the most of them wrong at once;
//! * `window_*`, over the terminal window (the recorded steps of the last tenth of the run): the
//!   evaluations, those with them all right, the least and the largest number of them wrong, and
//!   the wrong answers summed over the window among them and among all held-out equations;
//! * `final_*`: the wrong answers at the last step among them and among all;
//! * `tenths`, for every tenth of the run: its last recorded step with the wrong answers there
//!   among them and among all, then its evaluations and how many of these have one of them wrong.
//!
//! With `--list` it prints instead one line for every evaluation after that first one at which
//! one of them is wrong: the step, the wrong answers among them and among all held-out equations,
//! and the training errors and accuracy.
//!
//! It reads `heldout.txt`, `heldout_predictions.bin` and `history.jsonl`. Of the trainer it
//! shares the constants of the data alone, and it checks its reading of the predictions against
//! the trainer's own count of wrong held-out answers at every recorded step.

// The shared modules have items that only the trainer or its tests use.
#![allow(dead_code)]

#[path = "../data.rs"]
mod data;
#[path = "../mt.rs"]
mod mt;

use std::fmt::Display;
use std::path::Path;
use std::process::ExitCode;
use std::str::FromStr;

/// What a run recorded at an evaluation.
struct Record {
    step: u64,
    /// Wrong answers among the held-out equations with numerator zero.
    zero_wrong: usize,
    /// Wrong answers among all held-out equations.
    wrong: usize,
    train_errors: u64,
    train_acc: f64,
}

fn main() -> ExitCode {
    let mut runs: Vec<String> = std::env::args().skip(1).collect();
    let list = runs.first().is_some_and(|a| a == "--list");
    if list {
        runs.remove(0);
    }
    if runs.is_empty() || runs.iter().any(|a| a.starts_with("--")) {
        eprintln!("usage: numerator [--list] RUN_DIR...");
        return ExitCode::FAILURE;
    }
    for run in &runs {
        match load(Path::new(run)) {
            Ok((_, _, records)) if list => print!("{}", listing(run, &records)),
            Ok((heldout, zero, records)) => print!("{}", summary(run, heldout, zero, &records)),
            Err(message) => {
                eprintln!("numerator: {run}: {message}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}

/// The values of a numeric field in every line of `history.jsonl`.
fn field<T: FromStr>(text: &str, name: &str) -> Result<Vec<T>, String>
where
    T::Err: Display,
{
    let key = format!("\"{name}\":");
    text.lines()
        .map(|line| {
            let at = line.find(&key).ok_or(format!("a line of history.jsonl has no {name}"))?;
            let rest = line[at + key.len()..].chars();
            let number: String = rest.take_while(|c| c.is_ascii_digit() || "+-.eE".contains(*c)).collect();
            number.parse::<T>().map_err(|e| format!("{name}: {e}"))
        })
        .collect()
}

/// The held-out equations of `heldout.txt`: for each its numerator and the token of its answer.
fn equations(text: &str) -> Result<Vec<(u8, u8)>, String> {
    text.lines()
        .enumerate()
        .map(|(index, line)| {
            let fields: Vec<usize> = line
                .split_whitespace()
                .map(|f| f.parse::<usize>().map_err(|e| format!("heldout.txt line {}: {e}", index + 1)))
                .collect::<Result<_, _>>()?;
            match fields[..] {
                [num, den, quo] if num < data::P && den < data::P && quo < data::P && quo * den % data::P == num => {
                    Ok((num as u8, (data::NUM0 + quo) as u8))
                }
                _ => Err(format!("heldout.txt line {}: not an equation num / den = quo", index + 1)),
            }
        })
        .collect()
}

/// How many held-out equations a run has, how many of them have numerator zero, and its records.
fn load(run: &Path) -> Result<(usize, usize, Vec<Record>), String> {
    let read = |name: &str| std::fs::read(run.join(name)).map_err(|e| format!("{name}: {e}"));
    let text = |name: &str| String::from_utf8(read(name)?).map_err(|e| format!("{name}: {e}"));
    let equations = equations(&text("heldout.txt")?)?;
    let history = text("history.jsonl")?;
    let (steps, recorded): (Vec<u64>, Vec<u64>) = (field(&history, "step")?, field(&history, "heldout_errors")?);
    let (train_errors, train_acc): (Vec<u64>, Vec<f64>) =
        (field(&history, "train_errors")?, field(&history, "train_acc")?);
    let predictions = read("heldout_predictions.bin")?;
    if equations.is_empty() || predictions.len() != steps.len() * equations.len() {
        let (bytes, records) = (predictions.len(), steps.len());
        return Err(format!("{bytes} predictions for {records} steps of {} equations", equations.len()));
    }
    let mut records = Vec::with_capacity(steps.len());
    for (at, answers) in predictions.chunks_exact(equations.len()).enumerate() {
        let misses = |zero_only: bool| {
            let pairs = answers.iter().zip(&equations);
            pairs.filter(|(p, (num, answer))| (!zero_only || *num == 0) && *p != answer).count()
        };
        let (step, zero_wrong, wrong) = (steps[at], misses(true), misses(false));
        if wrong as u64 != recorded[at] {
            return Err(format!("step {step}: {wrong} wrong answers read, {} recorded", recorded[at]));
        }
        records.push(Record { step, zero_wrong, wrong, train_errors: train_errors[at], train_acc: train_acc[at] });
    }
    let zero = equations.iter().filter(|(num, _)| *num == 0).count();
    Ok((equations.len(), zero, records))
}

fn optional<T: Display>(value: Option<T>) -> String {
    value.map_or("null".to_string(), |v| v.to_string())
}

fn quoted(run: &str) -> String {
    format!("\"{}\"", run.replace('\\', "\\\\").replace('"', "\\\""))
}

/// The records after the first one at which every equation with numerator zero is right.
fn later(records: &[Record]) -> &[Record] {
    let first = records.iter().position(|r| r.zero_wrong == 0);
    first.map_or(&records[..0], |at| &records[at + 1..])
}

/// The line of a run, as the head of the file describes it.
fn summary(run: &str, heldout: usize, zero: usize, records: &[Record]) -> String {
    let count = |records: &[&Record]| records.len().to_string();
    let all: Vec<&Record> = records.iter().collect();
    let last = records.last().map_or(0, |r| r.step);
    let first = records.iter().find(|r| r.zero_wrong == 0);
    let sustained = records.iter().rposition(|r| r.zero_wrong != 0).map_or(0, |at| at + 1);
    let later: Vec<&Record> = later(records).iter().collect();
    let later_wrong: Vec<&Record> = later.iter().copied().filter(|r| r.zero_wrong != 0).collect();
    let unfitted = |records: &[&Record]| records.iter().filter(|r| r.train_errors != 0).count().to_string();
    let most = |records: &[&Record]| records.iter().map(|r| r.zero_wrong).max().unwrap_or(0).to_string();
    let window: Vec<&Record> = records.iter().filter(|r| r.step >= last - last / 10).collect();
    let right = |records: &[&Record]| records.iter().filter(|r| r.zero_wrong == 0).count().to_string();
    let fields = [
        ("heldout", heldout.to_string()),
        ("zero_numerator", zero.to_string()),
        ("evaluations", count(&all)),
        ("last_step", last.to_string()),
        ("zero_all_right_first", optional(first.map(|r| r.step))),
        ("heldout_right_then", optional(first.map(|r| heldout - r.wrong))),
        ("zero_all_right_sustained", optional(records.get(sustained).map(|r| r.step))),
        ("zero_all_right_evaluations", right(&all)),
        ("later_evaluations", count(&later)),
        ("later_unfitted", unfitted(&later)),
        ("later_zero_wrong", count(&later_wrong)),
        ("later_zero_wrong_unfitted", unfitted(&later_wrong)),
        ("later_zero_wrong_train_acc_max", optional(later_wrong.iter().map(|r| r.train_acc).reduce(f64::max))),
        ("later_zero_wrong_most", most(&later_wrong)),
        ("window_observations", count(&window)),
        ("window_zero_all_right", right(&window)),
        ("window_zero_wrong_min", window.iter().map(|r| r.zero_wrong).min().unwrap_or(0).to_string()),
        ("window_zero_wrong_max", most(&window)),
        ("window_zero_wrong_sum", window.iter().map(|r| r.zero_wrong).sum::<usize>().to_string()),
        ("window_wrong_sum", window.iter().map(|r| r.wrong).sum::<usize>().to_string()),
        ("final_zero_wrong", records.last().map_or(0, |r| r.zero_wrong).to_string()),
        ("final_wrong", records.last().map_or(0, |r| r.wrong).to_string()),
    ];
    let tenths: Vec<String> = (1..=10u64)
        .map(|tenth| {
            let inside = |r: &&Record| r.step * 10 > last * (tenth - 1) && r.step * 10 <= last * tenth;
            let block: Vec<&Record> = records.iter().filter(inside).collect();
            let end = block.last().map_or((0, 0, 0), |r| (r.step, r.zero_wrong, r.wrong));
            let wrong = block.iter().filter(|r| r.zero_wrong != 0).count();
            format!("[{},{},{},{},{wrong}]", end.0, end.1, end.2, block.len())
        })
        .collect();
    let body: Vec<String> = fields.iter().map(|(name, value)| format!("\"{name}\":{value}")).collect();
    format!("{{\"run\":{},{},\"tenths\":[{}]}}\n", quoted(run), body.join(","), tenths.join(","))
}

/// The lines of `--list`, as the head of the file describes them.
fn listing(run: &str, records: &[Record]) -> String {
    let wrong = later(records).iter().filter(|r| r.zero_wrong != 0);
    wrong
        .map(|r| {
            format!(
                "{{\"run\":{},\"step\":{},\"zero_wrong\":{},\"wrong\":{},\"train_errors\":{},\"train_acc\":{}}}\n",
                quoted(run),
                r.step,
                r.zero_wrong,
                r.wrong,
                r.train_errors,
                r.train_acc
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_and_equations_are_read() {
        let history = "{\"step\":0,\"train_acc\":0.25,\"heldout_errors\":2}\n\
                       {\"step\":10,\"lr\":0.5,\"train_acc\":1,\"train_acc_source\":0,\"heldout_errors\":0}\n";
        assert_eq!(field::<u64>(history, "step"), Ok(vec![0, 10]));
        assert_eq!(field::<u64>(history, "heldout_errors"), Ok(vec![2, 0]));
        assert_eq!(field::<f64>(history, "train_acc"), Ok(vec![0.25, 1.0]));
        assert!(field::<u64>(history, "seconds").is_err());
        assert!(field::<u64>(history, "train_acc").is_err());
        // 80 / 91 = 19 and 0 / 5 = 0 in the field of 97 elements; 1 / 2 is not 3.
        let read = equations("80 91 19\n0 5 0\n");
        assert_eq!(read, Ok(vec![(80, (data::NUM0 + 19) as u8), (0, data::NUM0 as u8)]));
        assert!(equations("1 2 3\n").is_err());
    }

    /// Ten held-out equations, two of them with numerator zero, recorded every ten steps to 100.
    #[test]
    fn a_run_is_summarized_and_listed() {
        let zero_wrong = [2, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0];
        let wrong = [10, 9, 8, 7, 9, 5, 4, 3, 2, 3, 1];
        let train_errors = [5, 4, 0, 0, 3, 0, 0, 0, 0, 0, 0];
        let records: Vec<Record> = (0..11)
            .map(|at| Record {
                step: 10 * at as u64,
                zero_wrong: zero_wrong[at],
                wrong: wrong[at],
                train_errors: train_errors[at],
                train_acc: 1.0 - train_errors[at] as f64 / 8.0,
            })
            .collect();
        let line = summary("runs/a \"b\"", 10, 2, &records);
        let expected = "{\"run\":\"runs/a \\\"b\\\"\",\"heldout\":10,\"zero_numerator\":2,\"evaluations\":11,\
            \"last_step\":100,\"zero_all_right_first\":20,\"heldout_right_then\":2,\
            \"zero_all_right_sustained\":100,\"zero_all_right_evaluations\":7,\"later_evaluations\":8,\
            \"later_unfitted\":1,\"later_zero_wrong\":2,\"later_zero_wrong_unfitted\":1,\
            \"later_zero_wrong_train_acc_max\":1,\"later_zero_wrong_most\":1,\"window_observations\":2,\
            \"window_zero_all_right\":1,\"window_zero_wrong_min\":0,\"window_zero_wrong_max\":1,\
            \"window_zero_wrong_sum\":1,\"window_wrong_sum\":4,\"final_zero_wrong\":0,\"final_wrong\":1,\
            \"tenths\":[[10,2,9,1,1],[20,0,8,1,0],[30,0,7,1,0],[40,1,9,1,1],[50,0,5,1,0],[60,0,4,1,0],\
            [70,0,3,1,0],[80,0,2,1,0],[90,1,3,1,1],[100,0,1,1,0]]}\n";
        assert_eq!(line, expected);
        let listed = "{\"run\":\"r\",\"step\":40,\"zero_wrong\":1,\"wrong\":9,\"train_errors\":3,\"train_acc\":0.625}\n\
            {\"run\":\"r\",\"step\":90,\"zero_wrong\":1,\"wrong\":3,\"train_errors\":0,\"train_acc\":1}\n";
        assert_eq!(listing("r", &records), listed);
        // A run that never has them all right has no later evaluation.
        assert_eq!(listing("r", &records[..2]), "");
        assert!(summary("r", 10, 2, &records[..2]).contains("\"zero_all_right_first\":null,"));
        assert!(summary("r", 10, 2, &[]).contains("\"evaluations\":0,\"last_step\":0,"));
    }
}
