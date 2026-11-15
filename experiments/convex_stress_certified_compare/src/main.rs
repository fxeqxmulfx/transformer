use convex_stress_certified_compare::{Result, runner};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd,output] if cmd=="forecast"=>runner::forecast_all(Path::new(output)),
        [cmd,root,producer,data,original] if cmd=="prepare"=>runner::prepare(Path::new(root),producer,Path::new(data),Path::new(original)),
        [cmd,source,root] if cmd=="repeat"=>runner::repeat_run(Path::new(source),Path::new(root)),
        [cmd,root,case] if cmd=="stage"=>runner::stage(Path::new(root),case),
        [cmd,root,case] if cmd=="score"=>runner::score(Path::new(root),case),
        [cmd,root,case] if cmd=="replay"=>runner::replay(Path::new(root),case),
        [cmd,root] if cmd=="summary"=>runner::summary(Path::new(root)),
        [cmd,a,b,out] if cmd=="verify"=>runner::verify_pair(Path::new(a),Path::new(b),Path::new(out)),
        _=>Err("usage: certified-compare forecast FRESH_JSON | prepare FRESH_ROOT COMMIT STREAM347_ROOT VERIFIED347 | repeat SOURCE FRESH_ROOT | {stage,score,replay} ROOT CASE | summary ROOT | verify ORIGINAL REPEAT FRESH_JSON".into())
    }
}
