//! The statistics catalogue: one entry per statistic, explaining what it
//! means, how it is computed and how to read it. This is the single source
//! of truth for `docs/STATISTICS.md`, the `--explain` text and the web
//! API, so a statistic cannot be shown without being explained. See
//! docs/superpowers/specs/2026-10-09-phase-12-statistics-skill-score-design.md.

use std::fmt::Write as _;

/// What a statistic is computed over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// One number for the whole run.
    Global,
    /// One number per strategy name.
    PerStrategy,
    /// One number per seat at the table.
    PerSeat,
    /// One number per table configuration.
    PerTable,
}

impl Scope {
    /// Every scope, in the order the generated docs list them.
    pub const ALL: [Scope; 4] = [
        Scope::Global,
        Scope::PerStrategy,
        Scope::PerSeat,
        Scope::PerTable,
    ];

    /// Heading used in the generated docs.
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            Scope::Global => "Whole run",
            Scope::PerStrategy => "Per strategy",
            Scope::PerSeat => "Per seat",
            Scope::PerTable => "Per table",
        }
    }
}

/// Plain-data description of one statistic.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct StatInfo {
    /// `snake_case`; equals the JSON field name where the value lives.
    pub id: &'static str,
    pub name: &'static str,
    pub scope: Scope,
    pub unit: &'static str,
    pub range: &'static str,
    /// One or two plain-language sentences.
    pub meaning: &'static str,
    /// The exact computation.
    pub formula: &'static str,
    /// Which direction is good, typical values, what would be suspicious.
    pub reading: &'static str,
    pub caveats: &'static str,
    /// Roadmap phase that introduced the statistic.
    pub since_phase: u32,
    /// Where the value appears in results.json.
    pub json_path: &'static str,
}

static CATALOG: &[StatInfo] = &[
    StatInfo {
        id: "role_counts_by_strategy",
        name: "Role counts",
        scope: Scope::PerStrategy,
        unit: "rounds",
        range: "0..rounds",
        meaning: "How often each strategy ended a round in each role (President, Vize, ..., Arschloch).",
        formula: "For every round of every match, each seat adds one to the count for its strategy's name and the role the seat finished with.",
        reading: "A strategy that is strong finishes in the top roles more often, so look at the share of President and Vize against the share of Arschloch. With equal strengths every role is about equally common (1/players each). Counts that are far from equal for identical strategies are suspicious and point at a seating or luck effect.",
        caveats: "Raw counts depend on how many rounds were played; compare shares, not totals. Single results are noisy and carry no error bars, see avg_rank and mean_role_score for numbers with standard errors.",
        since_phase: 1,
        json_path: "statistics.role_counts_by_strategy.<strategy>.<role>",
    },
    StatInfo {
        id: "voluntary_pass_rate",
        name: "Voluntary pass rate",
        scope: Scope::Global,
        unit: "share",
        range: "0..1",
        meaning: "How often players passed although they could have played a card. Passing when a legal play exists is a deliberate tactical choice.",
        formula: "sum of voluntary passes divided by sum of all passes over every match and seat; 0 if nobody ever passed.",
        reading: "Higher means more waiting and holding back. Typical values depend on the strategies; a greedy field is near 0, a patient field is clearly above it. Exactly 0 or 1 in a mixed field would be suspicious.",
        caveats: "Forced passes (no legal play) are excluded from the numerator, so the rate is not the same as the overall pass frequency. Depends on the pass rule in use. Confidence bounds are available as retention_interval.",
        since_phase: 2,
        json_path: "statistics.voluntary_pass_rate",
    },
    StatInfo {
        id: "voluntary_pass_rate_by_strategy",
        name: "Voluntary pass rate by strategy",
        scope: Scope::PerStrategy,
        unit: "share",
        range: "0..1",
        meaning: "The voluntary pass rate separately for each strategy: how patient or aggressive each one is.",
        formula: "For each strategy name: its voluntary passes divided by all its passes, over every match and seat it played.",
        reading: "Compare strategies with each other: a high value means the strategy often keeps cards back. It describes style, not strength; a strong strategy can be patient or aggressive.",
        caveats: "Strategies that rarely pass have few samples, so the rate is noisy. Seats using the same strategy type share one bucket.",
        since_phase: 2,
        json_path: "statistics.voluntary_pass_rate_by_strategy.<strategy>",
    },
    StatInfo {
        id: "role_retention_by_strategy",
        name: "Role retention",
        scope: Scope::PerStrategy,
        unit: "share",
        range: "0..1",
        meaning: "How often a strategy keeps the same role in the next round. The role from one round carries into the next through the card exchange, so this shows whether being on top tends to last.",
        formula: "For each strategy and role: of the round transitions where the strategy held the role and another round followed (held), the number where it held the same role again (retained_next_round); retention = retained_next_round / held.",
        reading: "A high President retention means the strategy defends its lead. By chance alone retention is about 1/players. A value much higher than that for President or Arschloch shows the exchange snowballs (the winner keeps the best cards); for a weak strategy high Arschloch retention is expected.",
        caveats: "Strategies that only played single-round matches have no entry. Counts for rare roles are small, use retention_interval for a 95% interval. Retention mixes skill and the advantage the exchange hands to the previous winner.",
        since_phase: 2,
        json_path: "statistics.role_retention_by_strategy.<strategy>.<role>.{held,retained_next_round}",
    },
    StatInfo {
        id: "first_round_placement_variance_by_strategy",
        name: "First-round placement variance",
        scope: Scope::PerStrategy,
        unit: "places squared",
        range: "0..unbounded",
        meaning: "How much a strategy's first-round result varies between matches that seated it identically. Since only the shuffle differs between those matches, this is a rough signal of how much luck the deal contributes.",
        formula: "Variance of the first-round placement (0 = the table's best role) across matches with the same seating (same strategy names per seat), computed per seat series and averaged. It is null when no seating appeared in at least 2 matches.",
        reading: "A strategy whose result barely depends on the deal has a small variance; a large variance means the result is dominated by luck. Compare with the variance of a uniformly random placement, which is the ceiling.",
        caveats: "Needs repeated identical seatings, so small batches give null. It counts all variation, including the strategy's own random choices, so it overstates pure card luck. luck_share and skill_score_duplicate measure luck more directly.",
        since_phase: 4,
        json_path: "statistics.first_round_placement_variance_by_strategy.<strategy>",
    },
    StatInfo {
        id: "matches_played",
        name: "Matches played",
        scope: Scope::Global,
        unit: "matches",
        range: "0..unbounded",
        meaning: "How many simulated matches the other numbers are based on.",
        formula: "The number of MatchResult records aggregated.",
        reading: "More matches means smaller standard errors; the error shrinks with the square root of the count (four times the matches halves it).",
        caveats: "Says nothing about the number of rounds per match; rounds are what most statistics count.",
        since_phase: 2,
        json_path: "statistics.matches_played",
    },
    StatInfo {
        id: "avg_rank",
        name: "Average finishing place",
        scope: Scope::PerStrategy,
        unit: "place",
        range: "1..players",
        meaning: "The average place a strategy finishes in, where 1 is the best (President) and the number of players is the worst (Arschloch). Reported per strategy and per seat, with a standard error.",
        formula: "Mean of the finishing place over every round of every match (place = index of the role in the table's role order, plus one). std_error = sample standard deviation of per-match mean place divided by the square root of the number of matches.",
        reading: "Lower is better. The average over a fair field is (players + 1) / 2, for example 2.5 at four players. A strategy more than about two standard errors away from that is credibly better or worse than average. Per seat values that differ from each other for equal strategies reveal a seating bias.",
        caveats: "Averages per match, because the rounds of one match are not independent (roles carry over). Places are not equally spaced in value; mean_role_score weights them linearly.",
        since_phase: 12,
        json_path: "extended.by_strategy.<name>.avg_rank (+ std_error), extended.by_seat.<n>.avg_rank",
    },
    StatInfo {
        id: "rank_distribution",
        name: "Rank distribution",
        scope: Scope::PerStrategy,
        unit: "percent of rounds",
        range: "0..100",
        meaning: "For each finishing place, the percentage of rounds in which a strategy finished there.",
        formula: "count of rounds finishing in place p divided by all rounds of the strategy, times 100. The percentages of one strategy sum to 100.",
        reading: "Shows the shape behind the average: two strategies with equal average place can differ in how often they win outright versus finish last. A fair field puts 100/players percent on each place.",
        caveats: "A distribution over many places needs many rounds to be smooth; small batches look jagged. No error bars.",
        since_phase: 12,
        json_path: "extended.by_strategy.<name>.rank_distribution",
    },
    StatInfo {
        id: "mean_role_score",
        name: "Mean role score",
        scope: Scope::PerStrategy,
        unit: "score",
        range: "-1..+1",
        meaning: "A single number for how well a strategy does: +1 means it is always President, -1 means it is always last, 0 is the middle. Reported per strategy and per seat, with a standard error.",
        formula: "Each finishing role maps linearly from +1 (best) to -1 (worst) with role_score(role, n) = 1 - 2 * place_index / (n - 1). The statistic is the mean over all rounds; std_error is computed over per-match means. This is also the fitness used in NEAT training.",
        reading: "Higher is better. A fair field averages exactly 0 (the scores of one table always sum to 0), so scores are relative to the opponents. Values near 0.1 are a small edge, above 0.3 is a clear one.",
        caveats: "Treats the gap between places as equal, which is a modelling choice. Depends on who the opponents are; use strength_rating to adjust for that.",
        since_phase: 12,
        json_path: "extended.by_strategy.<name>.mean_role_score (+ std_error), extended.by_seat.<n>.mean_role_score",
    },
    StatInfo {
        id: "strength_rating",
        name: "Strength rating",
        scope: Scope::PerStrategy,
        unit: "Elo-like points",
        range: "-unbounded..+unbounded",
        meaning: "How strong a strategy is relative to this particular field, adjusted for the opponents: finishing above a strong opponent counts for more than finishing above a weak one. The ratings average 0 over the field.",
        formula: "Bradley-Terry model fitted by the MM algorithm on every pair at every table in every round (who finished above whom), pooled by strategy name; strengths are converted to Elo-like points (400 * log10 of the strength ratio, centred at 0). std_error comes from 200 bootstrap resamples of matches.",
        reading: "Higher is better; 0 is the field average, a difference of 100 points means the stronger strategy finishes above the weaker one in about 64 percent of their meetings. Compare ratings within one run only.",
        caveats: "Only meaningful relative to this field: adding a weak opponent changes the others' numbers. A strategy that beat everyone in every game has an unbounded rating, so very large values show up as clipped. The bootstrap standard error is an estimate and needs many matches.",
        since_phase: 12,
        json_path: "extended.by_strategy.<name>.strength_rating (+ std_error)",
    },
    StatInfo {
        id: "retention_interval",
        name: "Wilson confidence interval",
        scope: Scope::PerStrategy,
        unit: "share",
        range: "0..1",
        meaning: "A range in which the true value of a rate (such as role retention or the voluntary pass rate) very likely lies, given how few or many cases it was measured on.",
        formula: "The Wilson score interval at 95 percent confidence: for k successes in n trials and z = 1.96, centre = (p + z^2/(2n)) / (1 + z^2/n), half-width = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n), with p = k / n.",
        reading: "Read it as the plausible range. If two intervals do not overlap, the rates probably differ. A wide interval means too few observations, run more matches.",
        caveats: "Assumes independent trials; consecutive rounds of one match are correlated, so the true uncertainty is somewhat larger than shown. Unlike the plain normal interval it stays inside 0..1 even for rare events.",
        since_phase: 12,
        json_path: "extended.*.<name>.<rate>_interval ([low, high])",
    },
    StatInfo {
        id: "useful_pass_share",
        name: "Useful pass share",
        scope: Scope::PerStrategy,
        unit: "share",
        range: "0..1",
        meaning: "Of the times a strategy passed on purpose, how often the pass was actually the better choice than the best card it could have played.",
        formula: "For a sample of voluntary passes the round is replayed K times from that exact state, once with the pass and once with each of the weakest and strongest legal play; a pass is useful if the average finishing place after passing is strictly better than after the best alternative. Share = useful passes / sampled voluntary passes.",
        reading: "Higher means the strategy passes for good reasons. A share near 0 means its passes cost it places; near 1 means passing is a real skill of this strategy. Compare with voluntary_pass_rate: many passes with a low useful share is wasteful.",
        caveats: "Estimated by rollouts, so noisy with small K or a small sample; deterministic for a fixed seed. The rollout opponents play their normal strategies, so the verdict is relative to them. Only two alternatives (weakest and strongest legal play) are compared, not every legal play. Off by default because it costs time.",
        since_phase: 12,
        json_path: "extended.by_strategy.<name>.useful_pass_share",
    },
    StatInfo {
        id: "useful_pass_gain",
        name: "Useful pass gain",
        scope: Scope::PerStrategy,
        unit: "places",
        range: "-players..+players",
        meaning: "On average, how many finishing places a voluntary pass gains or loses compared with the best alternative play.",
        formula: "Mean over sampled voluntary passes of (expected place after the best alternative minus expected place after passing), each estimated from K same-seed rollouts. Positive means passing was better.",
        reading: "Above 0 means the passes pay off on average; below 0 means the strategy would do better to just play. Gains are small, hundredths of a place are already notable.",
        caveats: "Same caveats as useful_pass_share: sampled, rollout noise, compares only the weakest and strongest legal play. Because the best alternative is chosen after the fact, a pass that is truly neutral still shows a slightly negative value.",
        since_phase: 12,
        json_path: "extended.by_strategy.<name>.useful_pass_gain",
    },
    StatInfo {
        id: "skill_score_duplicate",
        name: "Skill score (duplicate deals)",
        scope: Scope::PerStrategy,
        unit: "score",
        range: "-1..+1",
        meaning: "A luck-free skill measure: the same deals are replayed with the strategies rotated through every seat, so every strategy plays every hand once. The score is a strategy's mean role score over all those hands, plus or minus a standard error.",
        formula: "Groups of k matches (k = number of strategies) share one deal seed with the strategy assignment rotated through all seats. For strategy s in group g, x(s,g) is the mean role score over its k seatings; the skill score is the mean of x(s,g) over groups and the standard error is the standard deviation of x(s,g) over groups divided by the square root of the number of groups.",
        reading: "Higher is better; 0 is the field average and a fair field sums to 0. The standard error is much smaller than for ordinary games of the same size, so smaller differences are detectable. This is the reference the cheap estimator is compared with.",
        caveats: "Only round 1 is perfectly luck-free. Later rounds are luck-reduced, not luck-free, because the carried-over roles and exchanged cards depend partly on skill in earlier rounds and the rounds diverge between seatings. Needs a number of matches that is a multiple of the number of strategies. The deal comes from a separate random stream, so results differ from ordinary runs with the same seed.",
        since_phase: 12,
        json_path: "extended.skill.duplicate.<name>.{score,std_error}",
    },
    StatInfo {
        id: "skill_score_estimate",
        name: "Skill score (cheap estimate)",
        scope: Scope::PerStrategy,
        unit: "score",
        range: "-1..+1",
        meaning: "An inexpensive luck adjustment for ordinary runs: it predicts each seat's first-round result from the quality of the dealt hand and subtracts the part explained by luck.",
        formula: "Least squares regression of the round-1 role score of every seat on hand features (cards of Queen or higher, pairs, triples, quads, lowest card strength, sum of strengths) plus a constant per strategy. Adjusted score = actual score minus beta times (features minus their mean); the estimate is the mean adjusted score per strategy with a standard error.",
        reading: "Read like skill_score_duplicate: higher is better, 0 is average. It should land close to the duplicate result; estimator_agreement says how close. A big difference between the plain mean role score and this one means the strategy was helped or hurt by its hands.",
        caveats: "It can only remove the luck its features capture: luck that comes from how hands combine during play stays in. Uses round 1 only. The regression is linear and may be biased if the true hand effect is not. Costs almost nothing extra and works on ordinary runs.",
        since_phase: 12,
        json_path: "extended.skill.estimate.<name>.{score,std_error}",
    },
    StatInfo {
        id: "luck_share",
        name: "Luck share",
        scope: Scope::Global,
        unit: "share",
        range: "0..1",
        meaning: "How much of the spread of single-game results is down to card luck rather than play: 0 means skill decides everything, 1 means pure luck.",
        formula: "luck_share = 1 - 1 / variance_reduction, clamped to 0..1.",
        reading: "A card game with a lot of luck shows a high value. If it is large, many games are needed to separate two strategies, and duplicate deals pay off. Values near 0 would mean the deal hardly matters.",
        caveats: "A property of this field and rule set, not of one strategy. It rests on variance_reduction, which needs several duplicate groups to be stable. It counts only the luck that duplicate deals remove, so luck that arises during play (who happens to get which exchange cards in later rounds) is not included and the true luck share is somewhat higher.",
        since_phase: 12,
        json_path: "extended.skill.luck_share",
    },
    StatInfo {
        id: "variance_reduction",
        name: "Variance reduction",
        scope: Scope::Global,
        unit: "factor",
        range: "0..unbounded",
        meaning: "How many ordinary games one duplicate game is worth: a value of 5 means a duplicate group gives the accuracy of five times as many ordinary games.",
        formula: "Var(single-match score) / (k * Var(group mean)), where the single-match score is a strategy's mean role score in one match, k is the number of matches per group (number of strategies) and the group mean is x(s,g) as in skill_score_duplicate, pooled over strategies.",
        reading: "Larger is better for duplicate mode; 1 means no benefit. The cost is k times as many matches per group, so a value above 1 shows the format pays for itself, a value well above k is excellent.",
        caveats: "Estimated from the same sample, so itself noisy with few groups. Values below 1 can occur by chance.",
        since_phase: 12,
        json_path: "extended.skill.variance_reduction",
    },
    StatInfo {
        id: "estimator_agreement",
        name: "Estimator agreement",
        scope: Scope::Global,
        unit: "correlation / share",
        range: "-1..+1",
        meaning: "How closely the cheap estimator follows the duplicate result. High agreement means the cheap estimator can replace the expensive duplicate mode.",
        formula: "Spearman rank correlation of strategies ordered by skill_score_estimate and by skill_score_duplicate, together with the share of the duplicate variance reduction the estimator recovers; reported only when both modes ran.",
        reading: "Near 1 means the estimator gives the same ordering and removes a similar amount of luck, so duplicate deals can be skipped. Low values mean the estimator is not trustworthy for this game.",
        caveats: "With few strategies the rank correlation has few possible values and is easy to get by chance (three strategies give only a handful of orderings). Only available in the mode that runs both.",
        since_phase: 12,
        json_path: "extended.skill.estimator_agreement",
    },
];

/// Every statistic, existing and planned, in a stable order.
#[must_use]
pub fn catalog() -> &'static [StatInfo] {
    CATALOG
}

/// Looks a statistic up by id.
#[must_use]
pub fn find(id: &str) -> Option<&'static StatInfo> {
    CATALOG.iter().find(|s| s.id == id)
}

const INTRO: &str = "\
How to read these statistics. Every number comes from simulated play, so \
it is an estimate, not an exact truth. Where a value is shown as \
`x +- e`, `e` is the standard error: the typical amount the estimate would \
move if the whole experiment were repeated with different shuffles. \
Roughly, the true value lies within two standard errors of the estimate \
about 95 times out of 100, and a difference between two strategies smaller \
than two combined standard errors may well be luck. The standard error \
shrinks with the square root of the number of matches: four times as many \
matches halves it. Arschloch is a card game with a lot of luck in the deal, \
so small differences need many matches (or duplicate deals, see the skill \
statistics) before they mean anything. Scores are relative to the \
opponents in the run: a fair field averages out to exactly zero.";

/// Renders the full reference as Markdown (`docs/STATISTICS.md`). The
/// output is stable: no timestamps, catalogue order.
#[must_use]
pub fn render_markdown() -> String {
    let mut out = String::new();
    out.push_str("# Statistics reference\n\n");
    out.push_str(
        "<!-- Generated by `cargo run -p cli -- stats-doc > docs/STATISTICS.md`. \
         Do not edit by hand; edit sim/src/stats_catalog.rs. -->\n\n",
    );
    out.push_str(INTRO);
    out.push_str("\n\n## Contents\n\n");
    for scope in Scope::ALL {
        let members: Vec<&StatInfo> = CATALOG.iter().filter(|s| s.scope == scope).collect();
        if members.is_empty() {
            continue;
        }
        let _ = writeln!(out, "**{}**\n", scope.title());
        for s in members {
            let _ = writeln!(out, "- [{}](#{}) (`{}`)", s.name, s.id, s.id);
        }
        out.push('\n');
    }
    for s in CATALOG {
        let _ = write!(
            out,
            "## {name}\n\n<a id=\"{id}\"></a>`{id}` - {scope}, unit: {unit}, range: {range}, since phase {phase}\n\n\
             **Meaning.** {meaning}\n\n**Formula.** {formula}\n\n**Reading it.** {reading}\n\n\
             **Caveats.** {caveats}\n\n**JSON path.** `{path}`\n\n",
            name = s.name,
            id = s.id,
            scope = s.scope.title().to_lowercase(),
            unit = s.unit,
            range = s.range,
            phase = s.since_phase,
            meaning = s.meaning,
            formula = s.formula,
            reading = s.reading,
            caveats = s.caveats,
            path = s.json_path,
        );
    }
    out.truncate(out.trim_end().len());
    out.push('\n');
    out
}

/// Compact plain-text help for the `--explain` flag: one `name: meaning`
/// line per known id; unknown ids are skipped.
#[must_use]
pub fn render_explain(ids: &[&str]) -> String {
    let mut out = String::new();
    for s in ids.iter().filter_map(|id| find(id)) {
        let _ = writeln!(out, "{}: {}", s.name, s.meaning);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const REQUIRED_IDS: [&str; 18] = [
        "role_counts_by_strategy",
        "voluntary_pass_rate",
        "voluntary_pass_rate_by_strategy",
        "role_retention_by_strategy",
        "first_round_placement_variance_by_strategy",
        "matches_played",
        "avg_rank",
        "rank_distribution",
        "mean_role_score",
        "strength_rating",
        "retention_interval",
        "useful_pass_share",
        "useful_pass_gain",
        "skill_score_duplicate",
        "skill_score_estimate",
        "luck_share",
        "variance_reduction",
        "estimator_agreement",
    ];

    #[test]
    fn ids_are_unique_snake_case_and_cover_the_required_set() {
        let mut seen = HashSet::new();
        for s in catalog() {
            assert!(seen.insert(s.id), "duplicate id {}", s.id);
            assert!(
                s.id.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "{} is not snake_case",
                s.id
            );
        }
        for id in REQUIRED_IDS {
            assert!(find(id).is_some(), "missing {id}");
        }
    }

    #[test]
    fn no_field_is_empty() {
        for s in catalog() {
            for (field, text) in [
                ("id", s.id),
                ("name", s.name),
                ("unit", s.unit),
                ("range", s.range),
                ("meaning", s.meaning),
                ("formula", s.formula),
                ("reading", s.reading),
                ("caveats", s.caveats),
                ("json_path", s.json_path),
            ] {
                assert!(!text.trim().is_empty(), "{}: empty {field}", s.id);
            }
            assert!(s.since_phase > 0, "{}: since_phase", s.id);
        }
    }

    #[test]
    fn every_scope_is_listed_in_all_and_used_or_titled() {
        for scope in Scope::ALL {
            assert!(!scope.title().is_empty());
        }
        for s in catalog() {
            assert!(Scope::ALL.contains(&s.scope));
        }
    }

    #[test]
    fn scope_serialises_in_snake_case() {
        let json = serde_json::to_string(&Scope::PerStrategy).unwrap();
        assert_eq!(json, "\"per_strategy\"");
    }

    #[test]
    fn find_returns_the_entry_or_none() {
        assert_eq!(find("avg_rank").unwrap().id, "avg_rank");
        assert!(find("no_such_statistic").is_none());
    }

    #[test]
    fn markdown_mentions_every_id_and_name_and_is_stable() {
        let md = render_markdown();
        for s in catalog() {
            assert!(md.contains(s.id), "{}", s.id);
            assert!(md.contains(s.name), "{}", s.name);
        }
        assert_eq!(md, render_markdown());
        assert!(md.ends_with('\n') && !md.ends_with("\n\n"));
    }

    #[test]
    fn explain_skips_unknown_ids() {
        let text = render_explain(&["avg_rank", "bogus", "luck_share"]);
        assert_eq!(text.lines().count(), 2);
        assert!(text.starts_with("Average finishing place: "));
        assert!(render_explain(&["bogus"]).is_empty());
    }

    #[test]
    fn committed_docs_match_the_catalogue() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/STATISTICS.md");
        let committed = std::fs::read_to_string(path).expect("docs/STATISTICS.md exists");
        // Normalise line endings: git may check the file out with CRLF.
        assert!(
            committed.replace("\r\n", "\n") == render_markdown(),
            "docs/STATISTICS.md is stale; regenerate it with: \
             cargo run -p cli -- stats-doc > docs/STATISTICS.md"
        );
    }
}
