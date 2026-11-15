use convex_stress_frequency_programs::{Result, runner};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd,root,producer,data,original] if cmd=="prepare"=>runner::prepare(Path::new(root),producer,Path::new(data),Path::new(original)),
        [cmd,source,root] if cmd=="repeat"=>runner::repeat_run(Path::new(source),Path::new(root)),
        [cmd,root,case] if cmd=="stage"=>runner::stage(Path::new(root),case),
        [cmd,root,case] if cmd=="score"=>runner::score(Path::new(root),case),
        [cmd,root,case] if cmd=="replay"=>runner::replay(Path::new(root),case),
        [cmd,root] if cmd=="summary"=>runner::summary(Path::new(root)),
        _=>Err("usage: frequency-programs prepare FRESH_ROOT COMMIT DATA337 ORIGINAL35 | repeat SOURCE FRESH_ROOT | {stage,score,replay} ROOT CASE | summary ROOT".into()),
    }
}
