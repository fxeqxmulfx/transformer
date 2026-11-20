//! Refuse to build unless both pinned checkouts are at their revisions and clean.
use std::{path::Path, process::Command};

const PINS: [(&str, &str, &str); 2] = [
    (
        "../../.lake/reproduction/gpt",
        "5a37c515639f23037d191b0659a5a0853984328d",
        "research",
    ),
    (
        "../../.lake/reproduction/kgen",
        "a1b6bc2c956e1a5f35960e26b209e1b6c213096c",
        "crates",
    ),
];

fn git(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .expect("git runs")
}

fn main() {
    for (dir, revision, tree) in PINS {
        let dir = Path::new(dir);
        println!("cargo:rerun-if-changed={}", dir.join(tree).display());
        let head = git(dir, &["rev-parse", "HEAD"]);
        assert!(
            head.status.success(),
            "{} is missing: see README.md, Build and commands",
            dir.display()
        );
        assert_eq!(
            String::from_utf8(head.stdout).unwrap().trim(),
            revision,
            "{} is not at its pinned revision",
            dir.display()
        );
        let status = git(dir, &["status", "--porcelain", "--", tree]);
        assert!(
            status.status.success() && status.stdout.is_empty(),
            "{} has local changes under {tree}",
            dir.display()
        );
    }
}
