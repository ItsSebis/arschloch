//! Configuration surface for `Adaptive` (a later task in this module),
//! a single configurable strategy whose base behavior is `LowestLegal`
//! with five independently-toggleable modifiers layered on top:
//! card-counting, endgame denial (optionally sharpened by pass-based
//! hand-reading), deception, trick-lead
//! tempo, and lead-order bullying (all in docs/ROADMAP.md). Each modifier can be switched on or
//! off (and, for denial, chosen between two strengths) without writing
//! a new strategy struct, so batch runs can isolate which modifier
//! combination actually beats plain `LowestLegal`.
//!
//! This module only defines the configuration type and its `Display`/
//! `FromStr` grammar (for the CLI's per-seat strategy spec, added in a
//! later task) — the modifiers' actual algorithms live in their own
//! files/tasks and are untouched here.

use std::fmt;
use std::str::FromStr;

/// The `close` threshold `Adaptive` uses when denial is enabled and no
/// explicit `close=<n>` override is given. Matches
/// `EndgameDenial::CLOSE_TO_FINISHING` (`sim/src/strategies/
/// endgame_denial.rs`): 2 cards is deep into the final stretch at every
/// supported table size and deck variant (see that constant's own doc
/// comment for the full per-player-count rationale), so reusing it here
/// keeps `Adaptive(denial)`'s default trigger point consistent with the
/// standalone `EndgameDenial` strategy it's meant to generalize.
pub const DEFAULT_CLOSE: usize = 2; // matches EndgameDenial::CLOSE_TO_FINISHING

/// How the endgame-denial modifier is configured.
///
/// Denial switches this seat from conserving (`LowestLegal`-like) play
/// to control-retaining play once an opponent is judged close to
/// finishing. The two "on" variants differ only in *how* "close" is
/// judged; both share the same `close` threshold semantics as
/// `EndgameDenial::CLOSE_TO_FINISHING` (see `DEFAULT_CLOSE`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DenialMode {
    /// Denial is disabled; this modifier never changes `Adaptive`'s
    /// play relative to its other modifiers.
    Off,
    /// Judges "close to finishing" purely by hand size, exactly like
    /// the standalone `EndgameDenial` strategy: any active opponent at
    /// or below `close` cards triggers denial mode. Kept as a distinct
    /// variant (rather than always using hand-reading) so batch runs
    /// can compare the cheaper hand-size-only trigger against the
    /// pass-ceiling-aware one below.
    HandSize {
        /// Hand-size threshold at or below which an opponent counts as
        /// close to finishing.
        close: usize,
    },
    /// Sharpens the hand-size trigger with pass-based hand-reading: an
    /// opponent's pass history narrows the ceiling on what they can
    /// still beat, so this variant can judge an opponent "close" (or
    /// rule one out) using more than raw card count. The `close` field
    /// plays the same role as in `HandSize`.
    HandReading {
        /// Threshold applied on top of the pass-ceiling-derived signal;
        /// see `HandSize::close`.
        close: usize,
    },
}

/// Which independently-toggleable modifiers `Adaptive` layers on top of
/// its `LowestLegal` base behavior, and how each is tuned.
///
/// `AdaptiveConfig::NONE` (or `denial: DenialMode::Off` with
/// `counting: false` and `deception_rate: 0.0`) makes `Adaptive`'s
/// output byte-identical to plain `LowestLegal` for the same seed — no
/// modifier draws from the RNG unless it's enabled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveConfig {
    /// Enables the card-counting modifier: hold back a "precious"
    /// combo (one no unseen card can beat) instead of spending it
    /// immediately, the same signal `CardCounter` uses on its own.
    pub counting: bool,
    /// Enables (and tunes) the endgame-denial modifier: switch to
    /// aggressive, control-retaining play once an opponent is judged
    /// close to finishing. See `DenialMode` for the two ways
    /// "close" can be judged.
    pub denial: DenialMode,
    /// Probability in `[0.0, 1.0]` that this seat deceptively passes
    /// on a turn where it could legally beat the table, to make its
    /// hand harder for opponents to read. `0.0` disables deception
    /// entirely.
    pub deception_rate: f64,
    /// Enables the trick-lead-tempo modifier: once this seat's own hand
    /// is down to 2 cards or fewer, prefer the cheapest legal play
    /// that's provably safe against the whole active field over the
    /// base strategy's own cheapest-legal instinct, to seize the next
    /// trick's lead rather than risk losing it to an opponent's
    /// re-escalation. See `crate::strategies::adaptive::tempo` for the
    /// full rationale and the empirical data behind its fixed
    /// threshold (not configurable — see that module's doc comment).
    pub tempo: bool,
    /// Enables the lead-order-bullying modifier: while leading, with
    /// this hand shaped mostly as same-rank groups, lead the cheapest
    /// whole same-rank group before ever leading a single, banking
    /// singles for a free, unconditional finish later. See
    /// `crate::strategies::adaptive::bully` for the full rationale,
    /// including why — unlike `denial`'s `close` — this modifier has no
    /// opponent-proximity threshold at all (empirically found to only
    /// limit the benefit, never protect against a downside).
    pub bully: bool,
}

impl AdaptiveConfig {
    /// Every modifier disabled — `Adaptive::new(AdaptiveConfig::NONE)`
    /// behaves identically to plain `LowestLegal`.
    pub const NONE: Self = Self {
        counting: false,
        denial: DenialMode::Off,
        deception_rate: 0.0,
        tempo: false,
        bully: false,
    };

    /// Whether this configuration's fields are all in-range, in
    /// particular `deception_rate` being a finite value in `[0, 1]`.
    /// `Adaptive::new` asserts this holds; the CLI's parser only ever
    /// produces valid configurations via `FromStr`, so this is mainly a
    /// defensive check against configs built directly as struct
    /// literals.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.deception_rate.is_finite() && (0.0..=1.0).contains(&self.deception_rate)
    }
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            counting: true,
            denial: DenialMode::HandReading {
                close: DEFAULT_CLOSE,
            },
            deception_rate: 0.0,
            tempo: false,
            bully: false,
        }
    }
}

impl fmt::Display for AdaptiveConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut items: Vec<String> = Vec::new();
        if self.counting {
            items.push("counting".into());
        }
        let close_item = |c: usize| (c != DEFAULT_CLOSE).then(|| format!("close={c}"));
        match self.denial {
            DenialMode::Off => {}
            DenialMode::HandSize { close } => {
                items.push("denial".into());
                items.extend(close_item(close));
            }
            DenialMode::HandReading { close } => {
                items.push("reading".into());
                items.extend(close_item(close));
            }
        }
        if self.deception_rate > 0.0 {
            items.push(format!("deception={}", self.deception_rate));
        }
        if self.tempo {
            items.push("tempo".into());
        }
        if self.bully {
            items.push("bully".into());
        }
        if items.is_empty() {
            f.write_str("none")
        } else {
            f.write_str(&items.join(","))
        }
    }
}

/// A human-readable reason `str::parse::<AdaptiveConfig>()` failed —
/// e.g. an unknown option key, a duplicate key, a value out of range,
/// or `close` given without `denial`/`reading`. Carries just a message
/// (no structured variants) since the only consumer is the CLI, which
/// surfaces it as an error string to the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveConfigError(String);

impl fmt::Display for AdaptiveConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for AdaptiveConfigError {}

impl FromStr for AdaptiveConfig {
    type Err = AdaptiveConfigError;

    fn from_str(options: &str) -> Result<Self, Self::Err> {
        let err = |msg: String| AdaptiveConfigError(msg);
        if options.trim().is_empty() {
            return Err(err("empty option list (use `none` for no modifiers)".into()));
        }
        if options.trim() == "none" {
            return Ok(Self::NONE);
        }

        let mut counting = false;
        let mut reading = false;
        let mut denial = false;
        let mut close: Option<usize> = None;
        let mut deception_rate = 0.0f64;
        let mut tempo = false;
        let mut bully = false;
        let mut seen_keys: Vec<&str> = Vec::new();

        for item in options.split(',') {
            let item = item.trim();
            if item.is_empty() {
                return Err(err("empty option between commas".into()));
            }
            let (key, value) = match item.split_once('=') {
                Some((k, v)) => (k.trim(), Some(v.trim())),
                None => (item, None),
            };
            if seen_keys.contains(&key) {
                return Err(err(format!("option `{key}` given more than once")));
            }
            seen_keys.push(key);

            match (key, value) {
                ("counting", None) => counting = true,
                ("denial", None) => denial = true,
                ("reading", None) => reading = true,
                ("tempo", None) => tempo = true,
                ("bully", None) => bully = true,
                ("counting" | "denial" | "reading" | "tempo" | "bully", Some(_)) => {
                    return Err(err(format!("`{key}` doesn't take a value")));
                }
                ("close", Some(v)) => {
                    let parsed = v.parse::<usize>().map_err(|_| {
                        err(format!("`close` must be a whole number >= 1, got `{v}`"))
                    })?;
                    if parsed == 0 {
                        return Err(err("`close` must be >= 1".into()));
                    }
                    close = Some(parsed);
                }
                ("deception", Some(v)) => {
                    let rate: f64 = v
                        .parse()
                        .map_err(|_| err(format!("`deception` must be a number, got `{v}`")))?;
                    if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
                        return Err(err(format!("`deception` must be in [0, 1], got `{v}`")));
                    }
                    deception_rate = rate;
                }
                ("close" | "deception", None) => {
                    return Err(err(format!("`{key}` requires a value, e.g. `{key}=...`")));
                }
                (other, _) => {
                    return Err(err(format!(
                        "unknown option `{other}` (expected one of: counting, denial, \
                         reading, tempo, bully, deception=<rate>, close=<n>)"
                    )));
                }
            }
        }

        if close.is_some() && !denial && !reading {
            return Err(err(
                "`close` requires `denial` or `reading` to also be set".into()
            ));
        }

        let denial_mode = if reading {
            DenialMode::HandReading {
                close: close.unwrap_or(DEFAULT_CLOSE),
            }
        } else if denial {
            DenialMode::HandSize {
                close: close.unwrap_or(DEFAULT_CLOSE),
            }
        } else {
            DenialMode::Off
        };

        Ok(Self {
            counting,
            denial: denial_mode,
            deception_rate,
            tempo,
            bully,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_then_parse_round_trips_for_representative_configs() {
        let configs = [
            AdaptiveConfig::NONE,
            AdaptiveConfig::default(),
            AdaptiveConfig {
                counting: true,
                denial: DenialMode::Off,
                deception_rate: 0.0,
                tempo: false,
                bully: false,
            },
            AdaptiveConfig {
                counting: false,
                denial: DenialMode::HandSize { close: 2 },
                deception_rate: 0.0,
                tempo: false,
                bully: false,
            },
            AdaptiveConfig {
                counting: false,
                denial: DenialMode::HandSize { close: 3 },
                deception_rate: 0.0,
                tempo: false,
                bully: false,
            },
            AdaptiveConfig {
                counting: false,
                denial: DenialMode::HandReading { close: 2 },
                deception_rate: 0.0,
                tempo: false,
                bully: false,
            },
            AdaptiveConfig {
                counting: true,
                denial: DenialMode::Off,
                deception_rate: 0.2,
                tempo: false,
                bully: false,
            },
            AdaptiveConfig {
                counting: false,
                denial: DenialMode::Off,
                deception_rate: 0.0,
                tempo: true,
                bully: false,
            },
            AdaptiveConfig {
                counting: true,
                denial: DenialMode::HandReading { close: 2 },
                deception_rate: 0.1,
                tempo: true,
                bully: false,
            },
            AdaptiveConfig {
                counting: false,
                denial: DenialMode::Off,
                deception_rate: 0.0,
                tempo: false,
                bully: true,
            },
            AdaptiveConfig {
                counting: true,
                denial: DenialMode::HandReading { close: 3 },
                deception_rate: 0.1,
                tempo: true,
                bully: true,
            },
        ];
        for cfg in configs {
            let rendered = cfg.to_string();
            let parsed: AdaptiveConfig = rendered
                .parse()
                .unwrap_or_else(|e| panic!("failed to parse rendered config {rendered:?}: {e}"));
            assert_eq!(parsed, cfg, "round-trip mismatch for {rendered:?}");
        }
    }

    #[test]
    fn parses_none_and_bare_flags() {
        assert_eq!(
            "none".parse::<AdaptiveConfig>().unwrap(),
            AdaptiveConfig::NONE
        );
        assert_eq!(
            "counting".parse::<AdaptiveConfig>().unwrap(),
            AdaptiveConfig {
                counting: true,
                denial: DenialMode::Off,
                deception_rate: 0.0,
                tempo: false,
                bully: false,
            }
        );
    }

    #[test]
    fn parses_tempo() {
        assert_eq!(
            "tempo".parse::<AdaptiveConfig>().unwrap(),
            AdaptiveConfig {
                counting: false,
                denial: DenialMode::Off,
                deception_rate: 0.0,
                tempo: true,
                bully: false,
            }
        );
    }

    #[test]
    fn parses_tempo_combined_with_other_modifiers() {
        let cfg: AdaptiveConfig = "counting,reading,tempo".parse().unwrap();
        assert!(cfg.counting);
        assert!(matches!(cfg.denial, DenialMode::HandReading { .. }));
        assert!(cfg.tempo);
    }

    #[test]
    fn rejects_tempo_given_a_value() {
        assert!("tempo=1".parse::<AdaptiveConfig>().is_err());
    }

    #[test]
    fn parses_bully() {
        assert_eq!(
            "bully".parse::<AdaptiveConfig>().unwrap(),
            AdaptiveConfig {
                counting: false,
                denial: DenialMode::Off,
                deception_rate: 0.0,
                tempo: false,
                bully: true,
            }
        );
    }

    #[test]
    fn parses_bully_combined_with_other_modifiers() {
        let cfg: AdaptiveConfig = "counting,reading,tempo,bully".parse().unwrap();
        assert!(cfg.counting);
        assert!(matches!(cfg.denial, DenialMode::HandReading { .. }));
        assert!(cfg.tempo);
        assert!(cfg.bully);
    }

    #[test]
    fn rejects_bully_given_a_value() {
        assert!("bully=1".parse::<AdaptiveConfig>().is_err());
    }

    #[test]
    fn reading_implies_denial() {
        let cfg: AdaptiveConfig = "reading".parse().unwrap();
        assert!(matches!(cfg.denial, DenialMode::HandReading { .. }));
    }

    #[test]
    fn rejects_empty_option_list() {
        assert!("".parse::<AdaptiveConfig>().is_err());
    }

    #[test]
    fn rejects_duplicate_key() {
        assert!("counting,counting".parse::<AdaptiveConfig>().is_err());
    }

    #[test]
    fn rejects_unknown_key() {
        assert!("bogus".parse::<AdaptiveConfig>().is_err());
    }

    #[test]
    fn rejects_flag_given_a_value() {
        assert!("counting=1".parse::<AdaptiveConfig>().is_err());
    }

    #[test]
    fn rejects_parameter_given_no_value() {
        assert!("deception".parse::<AdaptiveConfig>().is_err());
        assert!("close".parse::<AdaptiveConfig>().is_err());
    }

    #[test]
    fn rejects_out_of_range_deception_rate() {
        assert!("deception=1.5".parse::<AdaptiveConfig>().is_err());
        assert!("deception=-0.1".parse::<AdaptiveConfig>().is_err());
        assert!("deception=nan".parse::<AdaptiveConfig>().is_err());
    }

    #[test]
    fn rejects_unparseable_numbers() {
        assert!("close=abc".parse::<AdaptiveConfig>().is_err());
        assert!("deception=abc".parse::<AdaptiveConfig>().is_err());
    }

    #[test]
    fn rejects_close_without_denial_or_reading() {
        assert!("close=3".parse::<AdaptiveConfig>().is_err());
        assert!("counting,close=3".parse::<AdaptiveConfig>().is_err());
    }

    #[test]
    fn rejects_close_of_zero() {
        assert!("denial,close=0".parse::<AdaptiveConfig>().is_err());
    }
}
