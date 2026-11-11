//! Run the frozen matrix in three seed streams, without overwriting evidence.
use std::{
    fs,
    path::PathBuf,
    process::{Command, Stdio},
};

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 1, "usage: suite OUT_ROOT");
    let root = PathBuf::from(&args[0]);
    fs::create_dir_all(&root).unwrap();
    let executable = std::env::current_exe().unwrap().with_file_name("convex-channels");
    assert!(executable.is_file(), "build all binaries before the suite");
    let success = std::thread::scope(|scope| {
        let mut streams = Vec::new();
        for seed in 0..3 {
            let root = &root;
            let executable = &executable;
            streams.push(scope.spawn(move || {
                for task in ["mqar", "recoded"] {
                    for method in ["channel", "copy", "transformer"] {
                        let dir = root.join(format!("{task}-{method}-{seed}"));
                        fs::create_dir(&dir).expect("output run already exists");
                        let log = fs::OpenOptions::new()
                            .create_new(true)
                            .write(true)
                            .open(dir.join("stdout.log"))
                            .unwrap();
                        println!("start {task} {method} seed {seed}");
                        let status = Command::new(executable)
                            .args(["run", task, method, &seed.to_string()])
                            .arg(&dir)
                            .stdout(Stdio::from(log.try_clone().unwrap()))
                            .stderr(Stdio::from(log))
                            .status()
                            .unwrap();
                        if !status.success() {
                            eprintln!(
                                "failed {task} {method} seed {seed}; evidence: {}",
                                dir.display()
                            );
                            return false;
                        }
                        println!("done {task} {method} seed {seed}");
                    }
                }
                true
            }));
        }
        streams.into_iter().map(|s| s.join().unwrap()).fold(true, |all, ok| all && ok)
    });
    assert!(success, "a run failed; keep its evidence and investigate");
}
