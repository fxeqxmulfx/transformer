//! Compile the pinned GPT data, scorer, schedule and kernel modules.
use std::{collections::BTreeSet, env, fs, path::Path, process::Command};

const REVISION: &str = "5a37c515639f23037d191b0659a5a0853984328d";

fn pin(dir: &Path, revision: &str, tree: &str) {
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .unwrap()
    };
    assert_eq!(
        String::from_utf8(git(&["rev-parse", "HEAD"]).stdout)
            .unwrap()
            .trim(),
        revision
    );
    assert!(
        git(&["diff", "--quiet", revision, "--", tree])
            .status
            .success()
    );
    println!("cargo:rerun-if-changed={}", dir.join(".git/HEAD").display());
    println!("cargo:rerun-if-changed={}", dir.join(tree).display());
}

// The pinned trainer lists only the budgets in its 35-task cohort. The
// larger authenticated GPT catalog declares additional horizons for the
// same update rule. Keep the pinned rate function and percentage cutoffs.
fn catalog_schedule(gpt: &Path, out: &Path) {
    let plan = fs::read(gpt.join("research/current_grokking/plan.json")).unwrap();
    let plan: serde_json::Value = serde_json::from_slice(&plan).unwrap();
    let budgets: BTreeSet<_> = plan["cases"]
        .as_array()
        .expect("missing GPT catalog cases")
        .iter()
        .map(|case| case["max_updates"].as_u64().expect("missing GPT budget"))
        .collect();
    assert!(
        [20_000, 30_000, 100_000]
            .iter()
            .all(|b| budgets.contains(b))
    );
    let budgets: Vec<_> = budgets.into_iter().collect();
    let source =
        fs::read_to_string(gpt.join("research/main_model/wsd_suite_native/src/schedule.rs"))
            .unwrap();
    let original = "[20_000, 30_000, 100_000]";
    assert_eq!(
        source.matches(original).count(),
        1,
        "unexpected pinned budget guard"
    );
    fs::write(
        out.join("schedule.rs"),
        source.replace(original, &format!("{budgets:?}")),
    )
    .unwrap();
}

fn main() {
    let lock = include_str!("Cargo.lock");
    let version = lock
        .split_once("name = \"kernite-cuda\"\nversion = \"")
        .and_then(|(_, rest)| rest.split_once('"'))
        .expect("missing kernite-cuda version in Cargo.lock")
        .0;
    println!("cargo:rustc-env=KERNITE_CUDA_VERSION={version}");
    let gpt = Path::new("../../.lake/reproduction/gpt")
        .canonicalize()
        .unwrap();
    pin(&gpt, REVISION, "research");
    let out = std::path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
    catalog_schedule(&gpt, &out);
    let kernels = gpt.join("research/main_model/phase_score_native/src");
    let suite = gpt.join("research/complex_geometry_suite/native/src");
    // The pinned stable-loss tests use the scalar oracle declared inline in
    // their binary. Include that same oracle when compiling test targets.
    let text = fs::read_to_string(kernels.join("main.rs")).unwrap();
    let marker = "#[cfg(test)]\nmod loss_precision {";
    let (_, tail) = text.split_once(marker).expect("missing pinned loss oracle");
    let (body, _) = tail
        .split_once("\n}\n")
        .expect("unterminated pinned loss oracle");
    let mut mods = format!("{marker}{body}\n}}\n");
    for (dir, names) in [
        (&kernels, "base_ops optimizer_kernels stable_loss"),
        (&suite, "blake data metrics pcg random"),
        (&out, "schedule"),
    ] {
        for name in names.split_whitespace() {
            let path = dir.join(format!("{name}.rs"));
            mods.push_str(&format!(
                "#[allow(dead_code, clippy::all)]\n#[path={path:?}] mod {name};\n"
            ));
        }
    }
    fs::write(out.join("upstream.rs"), mods).unwrap();
}
