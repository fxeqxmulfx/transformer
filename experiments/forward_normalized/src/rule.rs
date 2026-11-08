//! Only A's optimizer changes; all objectives and boundaries are reused.
use crate::config::Method;
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    Ce,
    Local,
    Normalized,
}
impl Rule {
    pub const ALL: [Self; 3] = [Self::Ce, Self::Local, Self::Normalized];
    pub fn name(self) -> &'static str {
        match self {
            Self::Ce => "ce-bp",
            Self::Local => "local-pair",
            Self::Normalized => "local-normalized-a",
        }
    }
    pub fn method(self) -> Method {
        if self == Self::Ce {
            Method::Ce
        } else {
            Method::Local
        }
    }
}
