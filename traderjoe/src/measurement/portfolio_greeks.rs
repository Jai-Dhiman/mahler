/// Static beta vs SPY. Quarterly human review — see GO_LIVE_GATE.md.
pub const BETA_TABLE: &[(&str, f64)] = &[
    ("SPY", 1.00),
    ("QQQ", 1.15),
    ("IWM", 1.25),
];

/// Look up beta for an underlying; returns 1.0 when not in table.
pub fn lookup_beta(underlying: &str) -> f64 {
    BETA_TABLE.iter()
        .find(|(u, _)| *u == underlying)
        .map(|(_, b)| *b)
        .unwrap_or(1.0)
}

/// Sum of (delta x beta) across underlyings.
pub fn beta_weighted_delta(contribs: &[(String, f64)]) -> f64 {
    contribs.iter().map(|(u, d)| d * lookup_beta(u)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_weighted_delta_applies_static_table() {
        // SPY delta 10, QQQ delta 5, IWM delta 4
        // beta-weighted = 10*1.00 + 5*1.15 + 4*1.25 = 10 + 5.75 + 5.0 = 20.75
        let contribs = vec![
            ("SPY".to_string(), 10.0),
            ("QQQ".to_string(), 5.0),
            ("IWM".to_string(), 4.0),
        ];
        let bwd = beta_weighted_delta(&contribs);
        assert!((bwd - 20.75).abs() < 1e-9);
    }

    #[test]
    fn unknown_underlying_uses_beta_1() {
        let contribs = vec![("UNKNOWN".to_string(), 3.0)];
        let bwd = beta_weighted_delta(&contribs);
        assert!((bwd - 3.0).abs() < 1e-9);
    }
}
