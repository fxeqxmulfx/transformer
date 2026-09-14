//! Print the canonical dump of the universal WASM machine's graph.
//!
//! Its counterpart on the Python side prints the same lines from the same
//! build, so `diff` is the whole test.

fn main() {
    let mg = alm_compile::interpreter::build();
    print!("{}", alm_compile::dump::dump(&mg));
}
