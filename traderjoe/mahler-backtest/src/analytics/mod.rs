//! Options analytics module.
//!
//! Provides:
//! - IV term structure analysis (contango/backwardation detection)
//! - Credit spread screening and candidate selection

pub mod iv_term_structure;
pub mod spread_screener;

pub use iv_term_structure::{IVTermStructure, TermStructureRegime, IVTermStructureAnalyzer};
pub use spread_screener::{SpreadCandidate, SpreadScreener, SpreadScreenerConfig};
