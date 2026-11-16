#[path = "pilot/mod.rs"]
mod pilot;

fn main() {
    if let Err(error) = pilot::run(std::env::args().skip(1).collect()) {
        eprintln!("pilot refused: {error}");
        std::process::exit(1);
    }
}
