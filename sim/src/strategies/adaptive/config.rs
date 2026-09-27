use std::fmt;
use std::str::FromStr;

pub const DEFAULT_CLOSE: usize = 2; // matches EndgameDenial::CLOSE_TO_FINISHING

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DenialMode {
    Off,
    HandSize { close: usize },
    HandReading { close: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveConfig {
    pub counting: bool,
    pub denial: DenialMode,
    pub deception_rate: f64,
}

impl AdaptiveConfig {
    pub const NONE: Self = Self {
        counting: false,
        denial: DenialMode::Off,
        deception_rate: 0.0,
    };

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
        if items.is_empty() {
            f.write_str("none")
        } else {
            f.write_str(&items.join(","))
        }
    }
}

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
                ("counting" | "denial" | "reading", Some(_)) => {
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
                         reading, deception=<rate>, close=<n>)"
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
            },
            AdaptiveConfig {
                counting: false,
                denial: DenialMode::HandSize { close: 2 },
                deception_rate: 0.0,
            },
            AdaptiveConfig {
                counting: false,
                denial: DenialMode::HandSize { close: 3 },
                deception_rate: 0.0,
            },
            AdaptiveConfig {
                counting: false,
                denial: DenialMode::HandReading { close: 2 },
                deception_rate: 0.0,
            },
            AdaptiveConfig {
                counting: true,
                denial: DenialMode::Off,
                deception_rate: 0.2,
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
                deception_rate: 0.0
            }
        );
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
