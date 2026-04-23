/// Assumed by backtest's ORATS slippage model: fills land ~34% into the bid-ask spread.
pub const ORATS_ASSUMED_SLIPPAGE_PCT_OF_SPREAD: f64 = 0.34;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SampleSizeTag {
    Insufficient,
    Weak,
    Ok,
}

impl SampleSizeTag {
    pub fn as_str(&self) -> &'static str {
        match self {
            SampleSizeTag::Insufficient => "INSUFFICIENT",
            SampleSizeTag::Weak => "WEAK",
            SampleSizeTag::Ok => "OK",
        }
    }
}

/// n<30: INSUFFICIENT (noise); 30<=n<100: WEAK (meaningful but not Kelly-confident); n>=100: OK.
pub fn sample_size_tag(n: usize) -> SampleSizeTag {
    if n < 30 {
        SampleSizeTag::Insufficient
    } else if n < 100 {
        SampleSizeTag::Weak
    } else {
        SampleSizeTag::Ok
    }
}

/// Paper fill violation: filled contracts exceed min of short/long leg displayed size.
/// A live exchange would not fill an order larger than displayed size; Alpaca paper does.
pub fn paper_fill_violation(contracts: i64, short_size: i64, long_size: i64) -> bool {
    contracts > short_size.min(long_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_size_tag_classifies_thresholds() {
        assert_eq!(sample_size_tag(0), SampleSizeTag::Insufficient);
        assert_eq!(sample_size_tag(29), SampleSizeTag::Insufficient);
        assert_eq!(sample_size_tag(30), SampleSizeTag::Weak);
        assert_eq!(sample_size_tag(99), SampleSizeTag::Weak);
        assert_eq!(sample_size_tag(100), SampleSizeTag::Ok);
        assert_eq!(sample_size_tag(1000), SampleSizeTag::Ok);
    }

    #[test]
    fn paper_fill_violation_flags_oversized_fill() {
        // 5 contracts vs displayed short 3 / long 10 -> violation (min = 3 < 5)
        assert!(paper_fill_violation(5, 3, 10));
        // 2 contracts vs displayed short 5 / long 5 -> no violation
        assert!(!paper_fill_violation(2, 5, 5));
    }
}
