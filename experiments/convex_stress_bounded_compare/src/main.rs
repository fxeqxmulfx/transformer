use convex_stress_bounded_compare::{Result, runner};
use std::path::Path;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let args: Vec<_> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["forecast",data,probe,summaries,producer]=>{
            let (inventory,_)=runner::inputs::plan(&runner::inputs::Sources {data:Path::new(data),probe:Path::new(probe),summaries:Path::new(summaries)},producer)?;
            println!("{}",serde_json::to_string(&inventory)?);
        }
        ["prepare",workspace,root,data,probe,summaries,producer]=>runner::inputs::prepare(
            &runner::inputs::Sources {data:Path::new(data),probe:Path::new(probe),summaries:Path::new(summaries)},
            Path::new(root),Path::new(workspace),producer)?,
        ["repeat",workspace,original,root]=>runner::inputs::repeat(Path::new(original),Path::new(root),Path::new(workspace))?,
        ["stage",workspace,root,name]=>runner::stage::run(Path::new(root),Path::new(workspace),name)?,
        ["seal",workspace,root]=>runner::stage::seal(Path::new(root),Path::new(workspace))?,
        ["score",workspace,root,name]=>runner::scoring::run(Path::new(root),Path::new(workspace),name,false)?,
        ["replay",workspace,root,name]=>runner::scoring::run(Path::new(root),Path::new(workspace),name,true)?,
        ["summary",workspace,root]=>runner::summary::run(Path::new(root),Path::new(workspace))?,
        ["verify",original,repeated,output]=>runner::summary::verify(Path::new(original),Path::new(repeated),Path::new(output))?,
        _=>return Err("expected forecast, prepare, repeat, stage, seal, score, replay, summary or verify with complete explicit paths".into()),
    }
    Ok(())
}
