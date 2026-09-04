use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Attack {
    Scan,
    FlashCrowd,
    PopularityShift,
    CostSpike,
    ExpensiveTail,
    HotKeyEmergence,
    HotKeyDecay,
    WorkingSetExplosion,
    MixedChaos,
}

impl Attack {
    pub fn as_str(self) -> &'static str {
        match self {
            Attack::Scan => "Scan",
            Attack::FlashCrowd => "FlashCrowd",
            Attack::PopularityShift => "PopularityShift",
            Attack::CostSpike => "CostSpike",
            Attack::ExpensiveTail => "ExpensiveTail",
            Attack::HotKeyEmergence => "HotKeyEmergence",
            Attack::HotKeyDecay => "HotKeyDecay",
            Attack::WorkingSetExplosion => "WorkingSetExplosion",
            Attack::MixedChaos => "MixedChaos",
        }
    }

    pub fn parse(s: &str) -> Option<Attack> {
        Some(match s {
            "Scan" | "scan" => Attack::Scan,
            "FlashCrowd" | "flash_crowd" => Attack::FlashCrowd,
            "PopularityShift" | "popularity_shift" => Attack::PopularityShift,
            "CostSpike" | "cost_spike" => Attack::CostSpike,
            "ExpensiveTail" | "expensive_tail" => Attack::ExpensiveTail,
            "HotKeyEmergence" | "hot_key_emergence" => Attack::HotKeyEmergence,
            "HotKeyDecay" | "hot_key_decay" => Attack::HotKeyDecay,
            "WorkingSetExplosion" | "working_set_explosion" => Attack::WorkingSetExplosion,
            "MixedChaos" | "mixed_chaos" => Attack::MixedChaos,
            _ => return None,
        })
    }

    pub fn description(self) -> &'static str {
        match self {
            Attack::Scan => "single-touch sweep over cold keys; nothing is worth admitting",
            Attack::FlashCrowd => "traffic collapses onto a handful of keys",
            Attack::PopularityShift => "the hot set is replaced by a different hot set",
            Attack::CostSpike => "regeneration gets an order of magnitude more expensive",
            Attack::ExpensiveTail => "rare keys carry the highest regeneration cost",
            Attack::HotKeyEmergence => "a cold key becomes the hottest key in the universe",
            Attack::HotKeyDecay => "the hottest key goes silent and must be released",
            Attack::WorkingSetExplosion => "the working set grows past capacity",
            Attack::MixedChaos => "several of the above overlap",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ActiveAttack {
    pub attack: Attack,
    pub started_s: f64,
    pub duration_s: f64,
}

impl ActiveAttack {
    pub fn is_live(&self, now_s: f64) -> bool {
        now_s >= self.started_s && now_s < self.started_s + self.duration_s
    }

    pub fn progress(&self, now_s: f64) -> f64 {
        if self.duration_s <= 0.0 {
            return 1.0;
        }
        ((now_s - self.started_s) / self.duration_s).clamp(0.0, 1.0)
    }
}
