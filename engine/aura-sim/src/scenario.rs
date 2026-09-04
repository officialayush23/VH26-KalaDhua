use serde::{Deserialize, Serialize};

use crate::attack::Attack;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scenario {
    SteadyZipf,
    FlashCrowd,
    ScanResistance,
    ExpensiveTail,
    ShiftingPopularity,
    MixedProduction,
}

impl Scenario {
    pub fn id(self) -> &'static str {
        match self {
            Scenario::SteadyZipf => "steady_zipf",
            Scenario::FlashCrowd => "flash_crowd",
            Scenario::ScanResistance => "scan_resistance",
            Scenario::ExpensiveTail => "expensive_tail",
            Scenario::ShiftingPopularity => "shifting_popularity",
            Scenario::MixedProduction => "mixed_production",
        }
    }

    pub fn parse(s: &str) -> Option<Scenario> {
        Some(match s {
            "steady_zipf" => Scenario::SteadyZipf,
            "flash_crowd" => Scenario::FlashCrowd,
            "scan_resistance" => Scenario::ScanResistance,
            "expensive_tail" => Scenario::ExpensiveTail,
            "shifting_popularity" => Scenario::ShiftingPopularity,
            "mixed_production" => Scenario::MixedProduction,
            _ => return None,
        })
    }

    pub const ALL: [Scenario; 6] = [
        Scenario::SteadyZipf,
        Scenario::FlashCrowd,
        Scenario::ScanResistance,
        Scenario::ExpensiveTail,
        Scenario::ShiftingPopularity,
        Scenario::MixedProduction,
    ];

    pub fn spec(self) -> ScenarioSpec {
        match self {
            Scenario::SteadyZipf => ScenarioSpec {
                id: self.id(),
                name: "Steady Zipf",
                description: "Stationary skewed traffic. The control case every policy should handle.",
                unique_keys: 60_000,
                zipf_alpha: 0.99,
                base_rps: 2_400.0,
                attacks: vec![],
            },
            Scenario::FlashCrowd => ScenarioSpec {
                id: self.id(),
                name: "Flash crowd",
                description: "Traffic collapses onto a few keys, then releases. Tests adaptation speed.",
                unique_keys: 80_000,
                zipf_alpha: 0.9,
                base_rps: 3_200.0,
                attacks: vec![Attack::FlashCrowd, Attack::HotKeyEmergence, Attack::HotKeyDecay],
            },
            Scenario::ScanResistance => ScenarioSpec {
                id: self.id(),
                name: "Scan resistance",
                description: "A sweep of one-touch keys tries to flush the working set. Admission has to refuse.",
                unique_keys: 400_000,
                zipf_alpha: 0.7,
                base_rps: 2_800.0,
                attacks: vec![Attack::Scan, Attack::WorkingSetExplosion],
            },
            Scenario::ExpensiveTail => ScenarioSpec {
                id: self.id(),
                name: "Expensive tail",
                description: "The rarest objects are the costliest to rebuild. Hit rate and cost disagree here.",
                unique_keys: 120_000,
                zipf_alpha: 0.85,
                base_rps: 2_000.0,
                attacks: vec![Attack::ExpensiveTail, Attack::CostSpike],
            },
            Scenario::ShiftingPopularity => ScenarioSpec {
                id: self.id(),
                name: "Shifting popularity",
                description: "The hot set is swapped out underneath the cache at intervals.",
                unique_keys: 150_000,
                zipf_alpha: 0.95,
                base_rps: 2_600.0,
                attacks: vec![Attack::PopularityShift],
            },
            Scenario::MixedProduction => ScenarioSpec {
                id: self.id(),
                name: "Mixed production",
                description: "Three applications with different cost shapes, plus overlapping disturbances.",
                unique_keys: 250_000,
                zipf_alpha: 0.92,
                base_rps: 4_000.0,
                attacks: vec![Attack::MixedChaos, Attack::FlashCrowd, Attack::Scan, Attack::CostSpike],
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ScenarioSpec {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub unique_keys: usize,
    pub zipf_alpha: f64,
    pub base_rps: f64,
    pub attacks: Vec<Attack>,
}
