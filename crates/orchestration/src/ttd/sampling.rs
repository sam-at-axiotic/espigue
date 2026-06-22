//! Per-trajectory sampling configuration for the TTD engine.
//! Source fidelity: consensus/diffusion/runner.py:276-301 + config.py:108-165.
//! StdRng (ChaCha12) is structurally faithful to Python random.Random(seed+i)
//! (Mersenne Twister) but NOT bit-identical — Phase 25 checks structural
//! diversity (N distinct temperatures), not bit parity (Research A1).

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::ttd::config::TtdConfig;

#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
}

/// Build N per-trajectory sampling configs from the master seed (runner.py:276-301).
/// randomize_sampling=false → N identical neutral configs (1.0/1.0/0), the Phase 23
/// reproduction default. randomize_sampling=true → per-trajectory StdRng::seed_from_u64(seed+i),
/// temperature from temp_range, top_p from top_p_range, top_k always 40 (runner.py:76).
pub fn build_sampling_configs(config: &TtdConfig) -> Vec<SamplingConfig> {
    let n = config.n_initial_drafts;
    if !config.randomize_sampling {
        return vec![SamplingConfig { temperature: 1.0, top_p: 1.0, top_k: 0 }; n];
    }
    let base = config.seed.unwrap_or(0);
    (0..n)
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(base.wrapping_add(i as u64));
            let temperature = rng.gen_range(config.temp_range.0..=config.temp_range.1);
            let top_p = rng.gen_range(config.top_p_range.0..=config.top_p_range.1);
            SamplingConfig { temperature, top_p, top_k: 40 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_temperatures_with_seed_42() {
        let mut cfg = TtdConfig::default();
        cfg.randomize_sampling = true;
        cfg.seed = Some(42);
        let configs = build_sampling_configs(&cfg);
        assert_eq!(configs.len(), 5);
        let temps: Vec<f32> = configs.iter().map(|c| c.temperature).collect();
        let unique: std::collections::HashSet<_> = temps.iter().map(|t| t.to_bits()).collect();
        assert_eq!(unique.len(), 5, "all N temperatures must be distinct");
    }

    #[test]
    fn per_trajectory_seed_reproducible() {
        let mut cfg = TtdConfig::default();
        cfg.randomize_sampling = true;
        cfg.seed = Some(42);
        let a = build_sampling_configs(&cfg);
        let b = build_sampling_configs(&cfg);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.temperature.to_bits(), y.temperature.to_bits());
        }
    }

    #[test]
    fn randomize_false_returns_neutral_configs() {
        let cfg = TtdConfig::default();
        let configs = build_sampling_configs(&cfg);
        assert!(configs.iter().all(|c| c.temperature == 1.0 && c.top_p == 1.0 && c.top_k == 0));
    }
}
