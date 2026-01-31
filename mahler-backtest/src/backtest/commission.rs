//! Commission model for options trading.
//!
//! Default: $1.00 per contract per leg (round-trip = $2.00 per contract per leg).

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Commission for a single transaction.
#[derive(Debug, Clone, Copy)]
pub struct Commission {
    pub contracts: i32,
    pub legs: usize,
    pub per_contract: Decimal,
    pub total: Decimal,
}

impl Commission {
    pub fn calculate(contracts: i32, legs: usize, per_contract: Decimal) -> Self {
        let total = per_contract * Decimal::from(contracts.abs()) * Decimal::from(legs as i32);
        Self {
            contracts,
            legs,
            per_contract,
            total,
        }
    }
}

/// Configurable commission model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommissionModel {
    /// Commission per contract per leg.
    pub per_contract: Decimal,
    /// Minimum commission per order.
    pub min_per_order: Decimal,
    /// Maximum commission per order (0 = unlimited).
    pub max_per_order: Decimal,
}

impl Default for CommissionModel {
    fn default() -> Self {
        Self {
            per_contract: Decimal::ONE,     // $1.00 per contract per leg
            min_per_order: Decimal::ZERO,   // No minimum
            max_per_order: Decimal::ZERO,   // No maximum
        }
    }
}

impl CommissionModel {
    /// Create a new commission model.
    pub fn new(per_contract: Decimal) -> Self {
        Self {
            per_contract,
            ..Default::default()
        }
    }

    /// Create a zero-commission model.
    pub fn zero() -> Self {
        Self {
            per_contract: Decimal::ZERO,
            min_per_order: Decimal::ZERO,
            max_per_order: Decimal::ZERO,
        }
    }

    /// Calculate commission for a trade.
    pub fn calculate(&self, contracts: i32, legs: usize) -> Commission {
        let mut commission = Commission::calculate(contracts, legs, self.per_contract);

        // Apply minimum
        if commission.total < self.min_per_order {
            commission.total = self.min_per_order;
        }

        // Apply maximum (if set)
        if self.max_per_order > Decimal::ZERO && commission.total > self.max_per_order {
            commission.total = self.max_per_order;
        }

        commission
    }

    /// Calculate round-trip commission (entry + exit).
    pub fn round_trip(&self, contracts: i32, legs: usize) -> Decimal {
        self.calculate(contracts, legs).total * Decimal::from(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_default_commission() {
        let model = CommissionModel::default();
        assert_eq!(model.per_contract, dec!(1));
    }

    #[test]
    fn test_commission_calculation() {
        let model = CommissionModel::default();

        // 10 contracts, 2 legs = $20
        let comm = model.calculate(10, 2);
        assert_eq!(comm.total, dec!(20));

        // 5 contracts, 1 leg = $5
        let comm = model.calculate(5, 1);
        assert_eq!(comm.total, dec!(5));
    }

    #[test]
    fn test_round_trip() {
        let model = CommissionModel::default();

        // 10 contracts, 2 legs, round-trip = $40
        let rt = model.round_trip(10, 2);
        assert_eq!(rt, dec!(40));
    }

    #[test]
    fn test_zero_commission() {
        let model = CommissionModel::zero();
        let comm = model.calculate(100, 4);
        assert_eq!(comm.total, dec!(0));
    }
}
