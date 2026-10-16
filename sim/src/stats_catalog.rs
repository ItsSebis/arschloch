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
    /// One number for a training run, from `cli analyse-run`.
    RunAnalysis,
}

impl Scope {
    /// Every scope, in the order the generated docs list them.
    pub const ALL: [Scope; 3] = [Scope::Global, Scope::PerStrategy, Scope::RunAnalysis];

    /// Heading used in the generated docs.
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            Scope::Global => "Whole run",
            Scope::PerStrategy => "Per strategy",
            Scope::RunAnalysis => "Run analysis (cli analyse-run)",
        }
    }
}

/// Plain-data description of one statistic.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct StatInfo {
    /// `snake_case`; the stable key of the statistic (it is not always
    /// the name of the JSON field, see `json_path`).
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
    /// Where the value appears in results.json: one or more paths
    /// separated by `"; "`. Segments are separated by `.`; `<name>` stands
    /// for every key of a JSON object (strategy name, role), `[]` for every
    /// element of an array and `{a,b}` for several fields at once. A test
    /// in the `cli` crate resolves every path against a real run. Entries of
    /// `Scope::RunAnalysis` point into one element of the `--json` array of
    /// `cli analyse-run` (`analysis.<field>...`).
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
        caveats: "Forced passes (no legal play) are excluded from the numerator, so the rate is not the same as the overall pass frequency. Depends on the pass rule in use. There is no confidence interval for this global rate; per-strategy intervals are available as voluntary_pass_rate_intervals.",
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
        formula: "Mean of the finishing place over every round of every match (place = index of the role in the table's role order, plus one). std_error = sample standard deviation of the series divided by the square root of the number of series, where a series is one strategy name in one match: the mean place of that name over its rounds, averaged over the seats it occupies in the match (so a name in two seats of a table counts once per match, not twice, because the seats of one name are negatively correlated). For a single-name batch n = number of matches. The value itself is weighted by rounds (every round of every seat counts once). Per-seat rows have one series per match.",
        reading: "Lower is better. The average over a fair field is (players + 1) / 2, for example 2.5 at four players. A strategy more than about two standard errors away from that is credibly better or worse than average. Per seat values that differ from each other for equal strategies reveal a seating bias.",
        caveats: "Averages per match, because the rounds of one match are not independent (roles carry over). Places are not equally spaced in value; mean_role_score weights them linearly.",
        since_phase: 12,
        json_path: "extended.by_strategy.<strategy>.avg_rank.{value,std_error}; extended.by_seat.[].avg_rank.{value,std_error}",
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
        json_path: "extended.by_strategy.<strategy>.rank_distribution",
    },
    StatInfo {
        id: "mean_role_score",
        name: "Mean role score",
        scope: Scope::PerStrategy,
        unit: "score",
        range: "-1..+1",
        meaning: "A single number for how well a strategy does: +1 means it is always President, -1 means it is always last, 0 is the middle. Reported per strategy and per seat, with a standard error.",
        formula: "Each finishing role maps linearly from +1 (best) to -1 (worst) with role_score(role, n) = 1 - 2 * place_index / (n - 1). The statistic is the mean over all rounds; std_error is computed over the same per-(match, strategy name) series as avg_rank (a name that holds several seats is averaged over them within the match). This is also the fitness used in NEAT training.",
        reading: "Higher is better. A fair field averages exactly 0 (the scores of one table always sum to 0), so scores are relative to the opponents. Values near 0.1 are a small edge, above 0.3 is a clear one.",
        caveats: "Treats the gap between places as equal, which is a modelling choice. Depends on who the opponents are; use strength_rating to adjust for that.",
        since_phase: 12,
        json_path: "extended.by_strategy.<strategy>.mean_role_score.{value,std_error}; extended.by_seat.[].mean_role_score.{value,std_error}",
    },
    StatInfo {
        id: "strength_rating",
        name: "Strength rating",
        scope: Scope::PerStrategy,
        unit: "Elo-like points",
        range: "-unbounded..+unbounded",
        meaning: "How strong a strategy is relative to this particular field, adjusted for the opponents: finishing above a strong opponent counts for more than finishing above a weak one. The ratings average 0 over the field.",
        formula: "Bradley-Terry model fitted by the MM algorithm on every pair at every table in every round (who finished above whom), pooled by strategy name; strengths are converted to Elo-like points (400 * log10 of the strength ratio, centred at 0). std_error comes from 200 bootstrap resamples of matches (null when resampling is switched off).",
        reading: "Higher is better; 0 is the field average, a difference of 100 points means the stronger strategy finishes above the weaker one in about 64 percent of their meetings. Compare ratings within one run only.",
        caveats: "Only meaningful relative to this field: adding a weak opponent changes the others' numbers. A strategy that beat everyone in every game would have an unbounded rating, but each strategy additionally plays one virtual average opponent (half a win, half a loss), which keeps every rating finite; nothing is clipped, and a very large value just means a very one-sided record. The bootstrap standard error is an estimate and needs many matches; with --bootstrap-resamples 0 (or 1) there is none and std_error is null, not 0.",
        since_phase: 12,
        json_path: "extended.by_strategy.<strategy>.strength_rating.{value,std_error}",
    },
    StatInfo {
        id: "retention_interval",
        name: "Wilson confidence interval",
        scope: Scope::PerStrategy,
        unit: "share",
        range: "0..1",
        meaning: "A range in which the true value of a rate (such as role retention; the voluntary pass rate has its own entry, voluntary_pass_rate_intervals) very likely lies, given how few or many cases it was measured on.",
        formula: "The Wilson score interval at 95 percent confidence: for k successes in n trials and z = 1.96, centre = (p + z^2/(2n)) / (1 + z^2/n), half-width = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n), with p = k / n.",
        reading: "Read it as the plausible range. If two intervals do not overlap, the rates probably differ. A wide interval means too few observations, run more matches.",
        caveats: "Assumes independent trials; consecutive rounds of one match are correlated, so the true uncertainty is somewhat larger than shown. Unlike the plain normal interval it stays inside 0..1 even for rare events.",
        since_phase: 12,
        json_path: "extended.role_retention_intervals.<strategy>.<role>.{low,high}",
    },
    StatInfo {
        id: "voluntary_pass_rate_intervals",
        name: "Voluntary pass rate interval",
        scope: Scope::PerStrategy,
        unit: "share",
        range: "0..1",
        meaning: "A 95 percent confidence interval for each strategy's voluntary pass rate (the per-strategy version of voluntary_pass_rate_by_strategy).",
        formula: "The Wilson score interval (see retention_interval) for k = the strategy's voluntary passes out of n = all its passes over every match and seat it played.",
        reading: "A wide interval means the strategy passed rarely, so its rate is poorly known. Two strategies whose intervals do not overlap probably differ in patience.",
        caveats: "Assumes independent passes; passes within one match are correlated, so the true uncertainty is somewhat larger. Only given per strategy: the global voluntary_pass_rate has no interval. Strategies that never passed have no interval.",
        since_phase: 12,
        json_path: "extended.voluntary_pass_rate_intervals.<strategy>.{low,high}",
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
        caveats: "Only round 1 is perfectly luck-free. Later rounds are luck-reduced, not luck-free, because the carried-over roles and exchanged cards depend partly on skill in earlier rounds and the rounds diverge between seatings. Needs a number of matches that is a multiple of the number of strategies. The deal comes from a separate random stream, so results differ from ordinary runs with the same seed. In the summary the plain score column equals the duplicate skill value by construction (the mean of the group means is the mean of the matches); its standard error is the counterfactual SE of an ordinary run of the same size, not an independent estimate (see plain_mean_role_score).",
        since_phase: 12,
        json_path: "extended.skill.duplicate.strategies.[].skill_score_duplicate.{value,std_error}; extended.skill.duplicate.strategies.[].skill_score_duplicate_round1.{value,std_error}",
    },
    StatInfo {
        id: "skill_score_estimate",
        name: "Skill score (cheap estimate)",
        scope: Scope::PerStrategy,
        unit: "score",
        range: "-1..+1",
        meaning: "An inexpensive luck adjustment for ordinary runs: it predicts each seat's first-round result from the quality of the dealt hand and subtracts the part explained by luck.",
        formula: "Least squares regression of the round-1 role score of every seat on hand features (cards of Queen or higher, pairs, triples, quads, lowest card strength, mean card strength, and hand size, because the uneven deal at 3, 5 and 6 players gives the first seats an extra card) plus a constant per strategy. Adjusted score = actual score minus beta times (features minus their mean); the estimate is the mean adjusted score per strategy with a standard error (seats of one name in a match are averaged first). The regression coefficients are reported under estimator_features.",
        reading: "Read like skill_score_duplicate: higher is better, 0 is average. It should land close to the duplicate result; estimator_agreement says how close. When both modes run, the estimate comes from a separate ordinary batch with different deals, so its standard error is honest and it differs from the duplicate value by noise. A big difference between the plain mean role score and this one means the strategy was helped or hurt by its hands.",
        caveats: "It can only remove the luck its features capture: luck that comes from how hands combine during play stays in. Uses round 1 only. With a single strategy name at the table there is nothing to adjust against, so the estimate is skipped (n/a). The regression is linear and may be biased if the true hand effect is not. Costs almost nothing extra and works on ordinary runs.",
        since_phase: 12,
        json_path: "extended.skill.estimate.strategies.[].skill_score_estimate.{value,std_error}",
    },
    StatInfo {
        id: "luck_share",
        name: "Luck share",
        scope: Scope::PerStrategy,
        unit: "share",
        range: "0..1",
        meaning: "How much of the spread of single-game results is down to card luck rather than play: 0 means skill decides everything, 1 means pure luck.",
        formula: "luck_share = 1 - 1 / variance_reduction, clamped to 0..1.",
        reading: "A card game with a lot of luck shows a high value. If it is large, many games are needed to separate two strategies, and duplicate deals pay off. Values near 0 would mean the deal hardly matters.",
        caveats: "Computed per strategy from the same deals, so the rows are similar; it describes this field and rule set rather than one strategy's quality. It rests on variance_reduction, which needs several duplicate groups to be stable. It counts only the luck that duplicate deals remove, so luck that arises during play (who happens to get which exchange cards in later rounds) is not included and the true luck share is somewhat higher.",
        since_phase: 12,
        json_path: "extended.skill.duplicate.strategies.[].luck_share; extended.skill.duplicate.strategies.[].luck_share_round1; extended.skill.estimate.strategies.[].luck_share",
    },
    StatInfo {
        id: "variance_reduction",
        name: "Variance reduction",
        scope: Scope::PerStrategy,
        unit: "factor",
        range: "0..unbounded",
        meaning: "How many ordinary games one duplicate game is worth: a value of 5 means a duplicate group gives the accuracy of five times as many ordinary games.",
        formula: "Var(single-match score) / (k * Var(group mean)), where the single-match score is a strategy's mean role score in one match, k is the number of matches per group (number of strategies) and the group mean is x(s,g) as in skill_score_duplicate. Computed for each strategy; the cheap estimator reports its own factor (the variance of the strategy's round-1 scores before and after the adjustment).",
        reading: "Larger is better; 1 means no benefit. The factor is already per match (the k matches of a group are accounted for in the formula), so any value above 1 means a duplicate or adjusted match is worth more than an ordinary one and the format pays off.",
        caveats: "Estimated from the same sample, so itself noisy with few groups. Values below 1 can occur by chance.",
        since_phase: 12,
        json_path: "extended.skill.duplicate.strategies.[].variance_reduction; extended.skill.duplicate.strategies.[].variance_reduction_round1; extended.skill.estimate.strategies.[].variance_reduction",
    },
    StatInfo {
        id: "estimator_agreement",
        name: "Estimator agreement",
        scope: Scope::Global,
        unit: "correlation / ratio",
        range: "rank_agreement -1..+1; mean_variance_removed_ratio 0..unbounded",
        meaning: "How closely the cheap estimator follows the duplicate result. High agreement means the cheap estimator can replace the expensive duplicate mode.",
        formula: "Kendall rank correlation (tau, rank_agreement, from -1 for reversed to +1 for identical order) of the strategies ordered by skill_score_estimate and by round-1 skill_score_duplicate, together with the mean share of the duplicate round-1 luck share the estimator recovers (mean_variance_removed_ratio, estimator luck share divided by duplicate luck share, which is at least 0 and can exceed 1 when the estimator's luck share is larger than the duplicate round-1 one); reported only when both modes ran.",
        reading: "Rank agreement near 1 and a removed ratio near 1 mean the estimator gives the same ordering and removes a similar amount of luck, so duplicate deals can be skipped. Low values mean the estimator is not trustworthy for this game.",
        caveats: "With few strategies the rank correlation has few possible values and is easy to get by chance (three strategies give only a handful of orderings). Only available in the mode that runs both, which plays about twice the matches: the duplicate groups and a separate ordinary batch (different deals) for the estimator. The estimate and the duplicate value therefore differ by sampling noise; gaps within about two standard errors are expected, perfect agreement would be suspicious.",
        since_phase: 12,
        json_path: "extended.skill.comparison.{rank_agreement,mean_variance_removed_ratio,verdict}",
    },
    StatInfo {
        id: "plain_mean_role_score",
        name: "Plain mean role score",
        scope: Scope::PerStrategy,
        unit: "score",
        range: "-1..+1",
        meaning: "The unadjusted mean role score next to the luck-adjusted skill scores, so you can see what the adjustment changed. It is the same number as mean_role_score, restricted to the matches and rounds the skill score covers.",
        formula: "Mean role score over all rounds of the duplicate matches (plain_mean_role_score), or over round 1 only (the _round1 variants, also used by the estimator). In duplicate mode the all-rounds plain score equals the duplicate skill value by construction (the mean of the group means is the mean of the matches); its standard error is the counterfactual SE of an ordinary run of the same size (sd over matches / sqrt(matches)), not an independent estimate.",
        reading: "Compare it with skill_score_duplicate or skill_score_estimate: the standard error shrinks while the value stays about the same, which is the point of the adjustment. A large shift in the estimator means the hands favoured or hurt that strategy.",
        caveats: "In duplicate mode do not compare it with the duplicate skill value as if they were two independent measurements: the values are equal and only the error bars differ.",
        since_phase: 12,
        json_path: "extended.skill.duplicate.strategies.[].plain_mean_role_score.{value,std_error}; extended.skill.duplicate.strategies.[].plain_mean_role_score_round1.{value,std_error}; extended.skill.estimate.strategies.[].plain_mean_role_score_round1.{value,std_error}",
    },
    StatInfo {
        id: "variance_explained",
        name: "Variance explained by the hands",
        scope: Scope::PerStrategy,
        unit: "share",
        range: "0..1",
        meaning: "How much of a strategy's round-1 seat-to-seat variation in role score the quality of the dealt hand accounts for.",
        formula: "Var(beta . x) / Var(y) over the strategy's seat observations, where y is the round-1 role score, x the hand features and beta the estimator's feature coefficients; clamped to 0..1.",
        reading: "High means the deal decides much of the first-round result for this strategy; low means the hand features barely matter (or the strategy dominates its field regardless of cards).",
        caveats: "Only the part of luck the features capture; it is not the luck share. Rows for different strategies come from the same regression, so they differ only through each strategy's own results.",
        since_phase: 12,
        json_path: "extended.skill.estimate.strategies.[].variance_explained",
    },
    StatInfo {
        id: "feature_r2",
        name: "Feature R-squared",
        scope: Scope::Global,
        unit: "share",
        range: "0..1",
        meaning: "How much extra first-round variation the hand features explain beyond knowing which strategy sat in the seat: the overall quality of the cheap estimator's regression.",
        formula: "Partial R^2 = 1 - SSR(full model) / SSR(strategy fixed effects only), where SSR is the residual sum of squares of the round-1 role score; clamped to 0..1.",
        reading: "Higher means the hands explain more of the luck, so the estimator removes more variance. Values around 0.1-0.3 are typical for this game; near 0 would mean the estimator does nothing.",
        caveats: "In-sample: it is computed on the same matches the regression was fitted on, so it is slightly optimistic with few matches.",
        since_phase: 12,
        json_path: "extended.skill.estimate.feature_r2",
    },
    StatInfo {
        id: "estimator_features",
        name: "Estimator features and coefficients",
        scope: Scope::Global,
        unit: "score per feature unit",
        range: "-unbounded..+unbounded",
        meaning: "The hand features the cheap estimator uses and how strongly each one moves the round-1 role score (the regression coefficients), plus how many matches the regression saw.",
        formula: "feature_names lists the features in order (high_cards, pairs, triples, quads, lowest_strength, mean_strength, hand_size); feature_coefficients holds one least-squares coefficient per feature, with the strategy fixed effects held separately. mean_strength is the mean (not the sum) of the card strengths of the hand; hand_size is the number of dealt cards. match_count is the number of matches with recorded hands.",
        reading: "A positive coefficient means a larger value of that feature helps the seat. Features are correlated, so read the coefficients together rather than one by one.",
        caveats: "When all hands have the same size hand_size is constant and its coefficient is arbitrary (it cannot be told apart from the strategy fixed effects) but harmless, because it never changes an adjustment. Coefficients are noisy with few matches.",
        since_phase: 12,
        json_path: "extended.skill.estimate.{match_count,feature_names,feature_coefficients}",
    },
    StatInfo {
        id: "difference_in_se",
        name: "Estimator gap in standard errors",
        scope: Scope::PerStrategy,
        unit: "standard errors",
        range: "-unbounded..+unbounded",
        meaning: "How far the cheap estimator's skill score lies from the duplicate result for one strategy, measured in combined standard errors.",
        formula: "(skill_score_estimate - skill_score_duplicate_round1) / sqrt(se_estimate^2 + se_duplicate^2); null (not defined) if both standard errors are 0.",
        reading: "Within about +-2 means the two agree up to noise; consistently large gaps for several strategies mean the estimator is biased for this game. The sign says whether the estimator is higher (+) or lower (-).",
        caveats: "In the both mode the two measurements use different deals, so a gap of a fraction of a standard error is expected; exactly 0 everywhere would be suspicious. With several strategies a few gaps beyond 2 happen by chance.",
        since_phase: 12,
        json_path: "extended.skill.comparison.strategies.[].{difference_in_se,variance_removed_ratio,skill_score_duplicate_round1,skill_score_estimate}",
    },
    StatInfo {
        id: "analysis_generations",
        name: "Generations analysed",
        scope: Scope::RunAnalysis,
        unit: "generations",
        range: "1..unbounded",
        meaning: "How many generation events the analysis is based on.",
        formula: "The number of generation events read from the run's events.jsonl (lines that are not events are not counted, see analysis_skipped_lines).",
        reading: "Everything else is a statement about this many points. Under about 30 generations the intervals are wide and the verdict is a rough hint only.",
        caveats: "A resumed run counts the generations of every part that is in the file (a generation that was run twice counts once). The stall and peak generations are generation numbers, not positions in this count; stall_index and peak_index are the positions.",
        since_phase: 17,
        json_path: "analysis.generations",
    },
    StatInfo {
        id: "analysis_best_generation",
        name: "Best generation",
        scope: Scope::RunAnalysis,
        unit: "generation number",
        range: "0..generations",
        meaning: "The generation whose champion the trainer kept as the best, chosen on the fixed re-evaluation set and confirmed on held-out matches.",
        formula: "Taken from the run_end event (best_generation); null when the run has no run_end event (it was stopped or is still running). The stop-early verdict then uses the generation with the best re-evaluation mean in the events instead, so a run still training is judged like the finished one (`cli train --when-flat`).",
        reading: "A best generation far before the end of the run means later generations added nothing the selection could see. Compare with the stall generation: the best generation is a single maximum, so it is usually later than the start of the plateau.",
        caveats: "It is the maximum of a noisy curve and therefore partly luck, see analysis_curse_reeval_minus_heldout. Not a measure of when the model was actually strongest.",
        since_phase: 17,
        json_path: "analysis.best_generation",
    },
    StatInfo {
        id: "analysis_final_reeval",
        name: "Final re-evaluation",
        scope: Scope::RunAnalysis,
        unit: "score",
        range: "-1..+1",
        meaning: "The best champion's score on the fixed re-evaluation set it was selected on, with its standard error.",
        formula: "mean and std_error of the best champion's mean role score over the fixed re-evaluation matches, from the run_end event (best_reeval); null without run_end.",
        reading: "Higher is better, 0 is the level of the pool. It is the score the trainer picked the champion by, so it is the optimistic one; use analysis_final_heldout for an unbiased figure.",
        caveats: "Selected as the maximum over many generations, so it is biased upwards (winner's curse). The standard error covers only the matches of this one measurement.",
        since_phase: 17,
        json_path: "analysis.final_reeval.{mean,std_error}",
    },
    StatInfo {
        id: "analysis_final_heldout",
        name: "Final held-out score",
        scope: Scope::RunAnalysis,
        unit: "score",
        range: "-1..+1",
        meaning: "The best champion's score on fresh matches it was not selected on, with its standard error.",
        formula: "mean and std_error of the best champion's mean role score over the held-out confirmation matches, from the run_end event (best_heldout); null for runs without held-out confirmation.",
        reading: "Higher is better. This is the honest estimate of the best model's level against the pool. A value clearly below the final re-evaluation shows the selection effect.",
        caveats: "Only one champion is confirmed this way, with the matches of one measurement: its standard error is not small. Absent in runs from before held-out confirmation existed.",
        since_phase: 17,
        json_path: "analysis.final_heldout.{mean,std_error}",
    },
    StatInfo {
        id: "analysis_peak_smoothed",
        name: "Peak of the smoothed curve",
        scope: Scope::RunAnalysis,
        unit: "score",
        range: "-1..+1",
        meaning: "The highest value of the champion re-evaluation curve after smoothing, the level the stall generation is measured against, and where it is reached.",
        formula: "The per-generation series is the champion's fixed-set re-evaluation score (the mean role score of the best genome of each generation against the fixed opponent pool on the same fixed matches). It is smoothed with a trailing mean over the last 10 generations (shorter at the start); peak_smoothed is the maximum of the smoothed series, peak_index the position of its first maximum and peak_generation the generation number there. final_smoothed is the smoothed value at the last generation, last_generation the number of the last generation.",
        reading: "Higher is better. The smoothed peak is lower than the best single generation because single maxima are partly luck. A peak in the last 5 percent of the run means the curve was still rising at the end (verdict still_improving); a final_smoothed more than one single-evaluation error below the peak means the run ended worse than it was (verdict declining).",
        caveats: "The window of 10 generations is fixed. A short run is smoothed over a large part of itself.",
        since_phase: 17,
        json_path: "analysis.peak_smoothed; analysis.peak_index; analysis.peak_generation; analysis.final_smoothed; analysis.last_generation",
    },
    StatInfo {
        id: "analysis_stall_generation",
        name: "Stall generation",
        scope: Scope::RunAnalysis,
        unit: "generation number",
        range: "first..last generation number",
        meaning: "The point from which the champion stopped getting better in a way the measurement can see: the first generation from which the smoothed re-evaluation score stays within one standard error of its peak.",
        formula: "Smooth the champion re-evaluation series with a trailing mean over 10 generations. stall_index is the first position i such that every smoothed value from i to the end is at least peak_smoothed - single_se, where single_se is analysis_noise_single_se (the mean standard error of one re-evaluation); stall_generation is the generation number at that position (generation numbers need not start at 0 or be gap free). If the last smoothed value itself is below that band no such position exists: stall_index is then the last position and stall_fallback is true (the fallback of docs/baselines/phase-13/analyse.py, whose index equals stall_index).",
        reading: "The generation after which the smoothed curve never leaves one noise unit of its peak. A stall in the first 60 percent of the run (by position) without a later best generation or a rising end means a long tail added nothing (verdict stop_early). stall_fallback true means the run ended below its peak, not that it was still improving: see the verdict declining.",
        caveats: "A band of one single-evaluation standard error is generous: the smoothed curve has less noise than one evaluation, so a real but slow improvement can hide inside the band and the stall is reported too early. A stall at the last generation says nothing about progress on its own (a falling curve gets it too): still_improving is decided by the position of the smoothed peak and the last-third trend, not by the stall.",
        since_phase: 17,
        json_path: "analysis.stall_generation; analysis.stall_index; analysis.stall_fallback",
    },
    StatInfo {
        id: "analysis_noise_single_se",
        name: "Noise: one evaluation",
        scope: Scope::RunAnalysis,
        unit: "score",
        range: "0..1",
        meaning: "How accurately one champion re-evaluation measures the score: the typical standard error of a single generation's number.",
        formula: "Mean over generations of the standard error of the champion's re-evaluation mean (std_error of champion.reeval).",
        reading: "Compare with the changes between generations and with the trend over the run: improvements smaller than this per generation cannot be seen in single evaluations. More --reeval-matches or duplicate-deal scoring make it smaller.",
        caveats: "It covers the dealing luck of the evaluation matches only, not the luck of which genome the evolution happened to produce.",
        since_phase: 17,
        json_path: "analysis.noise.single_se",
    },
    StatInfo {
        id: "analysis_noise_change_sd",
        name: "Noise: generation-to-generation change",
        scope: Scope::RunAnalysis,
        unit: "score",
        range: "0..unbounded",
        meaning: "How much the champion's re-evaluation score jumps from one generation to the next.",
        formula: "Sample standard deviation (n - 1 divisor) of the differences re-evaluation mean(g+1) - re-evaluation mean(g) over consecutive generations; null below two differences (three generations).",
        reading: "If it is more than twice the single-evaluation standard error, the curve moves by more than the measurement noise explains, which is the sign of selection noise (a different champion each generation) rather than steady progress. The noise_limited verdict uses this comparison (strictly greater, and only with a single-evaluation standard error above 0).",
        caveats: "Differences of a rising curve include the rise, so a fast steady climb also raises it; read it together with the trend.",
        since_phase: 17,
        json_path: "analysis.noise.change_sd",
    },
    StatInfo {
        id: "analysis_trend_overall",
        name: "Trend over the whole run",
        scope: Scope::RunAnalysis,
        unit: "score per 100 generations",
        range: "-unbounded..+unbounded",
        meaning: "How fast the champion's re-evaluation score changed on average over the whole run, with a 95 percent interval.",
        formula: "Ordinary least-squares slope of the champion's fixed-set re-evaluation score (the mean role score of the best genome of each generation against the fixed opponent pool on the same fixed matches) on the generation number, times 100. The 95 percent interval is slope +- t * standard error of the slope, with the t quantile for n - 2 degrees of freedom (table up to 30, then interpolated linearly in 1/df between 30, 40, 60, 120 and the normal 1.96 at infinity), and the standard error is corrected for lag-1 autocorrelation: with r the lag-1 autocorrelation of the residuals of the fit, when r > 0 it is multiplied by sqrt((1 + r) / (1 - r)), at most 5. Null below three generations.",
        reading: "Positive is improving. If the interval contains 0 the run shows no trend it can tell from luck. Early progress usually makes the whole-run trend positive even when the end is flat, so look at the last third as well.",
        caveats: "Generations are not independent (the champion of one generation is related to the next). The interval is corrected for lag-1 autocorrelation of the residuals (the correction of an AR(1) process, estimated from the data; approximate, and noisy for few points), which widens it but may still be too narrow for strongly structured curves. A straight line is a rough description of a curve that rises fast and then flattens.",
        since_phase: 17,
        json_path: "analysis.trend.overall.{value,low,high}",
    },
    StatInfo {
        id: "analysis_trend_last_third",
        name: "Trend over the last third",
        scope: Scope::RunAnalysis,
        unit: "score per 100 generations",
        range: "-unbounded..+unbounded",
        meaning: "The same trend, fitted only to the last third of the run: is the champion still getting better at the end?",
        formula: "As analysis_trend_overall (including the interval corrected for lag-1 autocorrelation), over the last third of the generations (the last max(1, n div 3) of them). A trend counts as rising when the whole interval is above 0, falling when it is below 0 and flat when it contains 0. Null with fewer than three generations in the third, so runs shorter than nine generations have none.",
        reading: "The key number of the verdicts: flat means the run has reached its plateau, rising means it should be continued, falling means it got worse (verdict declining). A flat interval that is very wide says only that the data cannot tell.",
        caveats: "Fewer points than the whole run, so a wide interval; the interval is corrected for autocorrelation, approximately. Flat is not the same as proven zero.",
        since_phase: 17,
        json_path: "analysis.trend.last_third.{value,low,high}",
    },
    StatInfo {
        id: "analysis_curse_reeval_minus_heldout",
        name: "Winner's curse: re-evaluation minus held-out",
        scope: Scope::RunAnalysis,
        unit: "score",
        range: "-unbounded..+unbounded",
        meaning: "How much better the best champion looked on the matches it was selected on than on fresh matches. A positive gap is the selection effect: picking the maximum of many noisy numbers picks some luck with it.",
        formula: "best_reeval.mean - best_heldout.mean from the run_end event (value); in_heldout_se is the same divided by the held-out standard error, in_combined_se divided by sqrt(se_reeval^2 + se_heldout^2), the standard error of the difference of the two measurements (each null when that error is 0). Null without run_end or without a held-out score.",
        reading: "Around 0 means the best generation is as good as it looked. A gap of two combined standard errors or more means a large part of the best generation's lead is luck, so it is not evidence of progress. The noise_limited verdict uses in_combined_se >= 2 (the held-out standard error alone is not used: with a maximum selected from many generations a gap of one is almost always there).",
        caveats: "A selection effect, not a defect of the model: it exists whenever a maximum is selected from a noisy series, and it is expected to be positive on average. A single run's gap is itself noisy; look at the gaps of several seeds.",
        since_phase: 17,
        json_path: "analysis.curse.reeval_minus_heldout.{value,in_heldout_se,in_combined_se}",
    },
    StatInfo {
        id: "analysis_curse_train_minus_reeval",
        name: "Training fitness minus re-evaluation",
        scope: Scope::RunAnalysis,
        unit: "score",
        range: "-unbounded..+unbounded",
        meaning: "How much higher the champion's training fitness is than its re-evaluation score, on average: how optimistic the numbers the evolution selects on are.",
        formula: "Mean over generations of champion.train_fitness - champion.reeval.mean. The training fitness comes from the few matches the genome played in training, the re-evaluation from the fixed set.",
        reading: "Close to 0 means training fitness is a fair guide. A large positive value means the champion of each generation is picked on luck, which is the reason for the fixed re-evaluation set and for more matches per genome.",
        caveats: "Depends on the fitness variant and the training opponents: with racing, the fitness-sharing or skill-weighted variants the two numbers do not measure the same thing exactly.",
        since_phase: 17,
        json_path: "analysis.curse.train_minus_reeval",
    },
    StatInfo {
        id: "analysis_opponent_means",
        name: "Per opponent: first and last third",
        scope: Scope::RunAnalysis,
        unit: "score",
        range: "-1..+1",
        meaning: "The champion's score against one fixed opponent alone, averaged over the first and over the last third of the run: the same windows as the champion's own last-third trend.",
        formula: "For each opponent name, the mean of champion.opponents[name].score.mean over the generations of the first third and of the last third of the run (the last max(1, n div 3) generations, like analysis_trend_last_third) that carry that opponent. Null when no generation of the window carries it.",
        reading: "A last-third mean far above the first third means the champion learned to beat that opponent. The difference between opponents shows where the remaining headroom is (a score near +1 leaves little).",
        caveats: "Averages of noisy numbers; use the trend and its class for statements about direction. The pool of an old run may change between parts, so only generations that report the opponent count.",
        since_phase: 17,
        json_path: "analysis.opponents.[].{name,first_third_mean,last_third_mean}",
    },
    StatInfo {
        id: "analysis_opponent_trend",
        name: "Per opponent: trends and class",
        scope: Scope::RunAnalysis,
        unit: "score per 100 generations",
        range: "-unbounded..+unbounded",
        meaning: "How the champion's score against one opponent changed over the run, and whether that change is rising, flat or falling.",
        formula: "last_third_slope: the slope with its 95 percent interval (value, low, high), computed as analysis_trend_last_third (interval corrected for lag-1 autocorrelation) over the generations of the last third of the run that report this opponent; null below three points. overall_slope: the same over every generation that reports it. class is taken from last_third_slope: rising if its interval is above 0, falling if below 0, flat if it contains 0; null if there is no slope.",
        reading: "The class describes the same window as the champion's last-third trend, so the verdict pool_limited (rising against some opponents and flat against others while the champion is flat) says what the last third measured. overall_slope is the whole-run slope for comparison: an opponent that rose early and is flat now is flat here.",
        caveats: "Flat means not distinguishable from 0, not that the champion cannot improve. A score already near +1 cannot rise: such an opponent is flat because it is beaten, which says it has nothing left to teach.",
        since_phase: 17,
        json_path: "analysis.opponents.[].last_third_slope.{value,low,high}; analysis.opponents.[].overall_slope.{value,low,high}; analysis.opponents.[].class",
    },
    StatInfo {
        id: "analysis_growth",
        name: "Network growth",
        scope: Scope::RunAnalysis,
        unit: "nodes, connections",
        range: "0..unbounded",
        meaning: "How large the champion's network was in the first and the last generation, and whether it kept growing without getting better.",
        formula: "first/last_hidden_nodes and first/last_connections: the hidden nodes and enabled connections of the champion in the first and last generation. third_start_nodes and third_end_nodes: the mean hidden-node count of the first and of the last third_window generations of the last third (third_window = the third's length div 2, between 1 and 3). without_gain is true if third_end_nodes > third_start_nodes and the last-third trend interval contains 0.",
        reading: "Growth together with a rising score is the normal case. without_gain means complexity is added that does not pay; see the verdict growth_without_gain.",
        caveats: "Counts the champion of each generation, not the whole population. Means over up to three generations make a node gained or lost in a single generation count less; a short run (third under six generations) uses fewer generations per mean.",
        since_phase: 17,
        json_path: "analysis.growth.{first_hidden_nodes,last_hidden_nodes,first_connections,last_connections,third_start_nodes,third_end_nodes,third_window,without_gain}",
    },
    StatInfo {
        id: "analysis_population_top_bin_share",
        name: "Population convergence",
        scope: Scope::RunAnalysis,
        unit: "share",
        range: "0..1",
        meaning: "How much of the population sits in the best fitness bin: a high value means nearly all genomes are equally good and the evolution has little variety left.",
        formula: "For each generation of the last third: genomes in the top bin of the fitness histogram divided by all genomes in the histogram; the mean of that share over those generations. Null without histograms.",
        reading: "Low values (about 10 percent) mean a spread population. 50 percent or more triggers the converged verdict.",
        caveats: "Depends on the binning of the fitness histogram, which spans the generation's own fitness range, so a single outlier genome also shifts the shares. Equal fitness is also a sign of fitness saturating at a ceiling.",
        since_phase: 17,
        json_path: "analysis.population.top_bin_share",
    },
    StatInfo {
        id: "analysis_population_species",
        name: "Species and compatibility threshold",
        scope: Scope::RunAnalysis,
        unit: "species, distance",
        range: "species 1..population",
        meaning: "The number of species and the compatibility threshold in the first and the last generation.",
        formula: "species_first / species_last: the number of species in the first and last generation. threshold_first / threshold_last: the compatibility distance below which two genomes count as one species, as adjusted by the trainer to reach its target species count.",
        reading: "A threshold that fell a long way (for example 0.5 to 0.1) means the population became very similar and the trainer had to make the species finer to keep the target count; one that rose means genomes diverged.",
        caveats: "Descriptive only: no verdict rule uses these numbers. They say little when the threshold is not adapted.",
        since_phase: 17,
        json_path: "analysis.population.{species_first,species_last,threshold_first,threshold_last}",
    },
    StatInfo {
        id: "analysis_racing",
        name: "Racing totals",
        scope: Scope::RunAnalysis,
        unit: "matches, share, score",
        range: "0..unbounded",
        meaning: "What racing (extra matches for the leading genomes) cost and how much it shifted the fitness values, summed over the run. Present only for runs with racing.",
        formula: "generations: generations with racing data. extra_matches: sum over those generations of raced_final (or raced if larger) times the extra matches per raced genome. extra_share: extra_matches / (extra_matches + generations * population_size * matches_per_genome); null without the run's start event. mean_shift: mean over generations of the mean fitness change of the raced genomes between before and after pooling.",
        reading: "extra_share is the price of racing in simulation work. A clearly negative mean_shift means the leaders' first numbers were too high, which is the selection effect racing exists to cancel.",
        caveats: "The share counts matches, not time, and uses the configured population and matches per genome of the start event, so it is off for runs whose settings changed on resume.",
        since_phase: 17,
        json_path: "analysis.racing.{generations,extra_matches,extra_share,mean_shift}",
    },
    StatInfo {
        id: "analysis_timings",
        name: "Seconds per stage",
        scope: Scope::RunAnalysis,
        unit: "seconds",
        range: "0..unbounded",
        meaning: "Where the run spent its time, summed over the run: training evaluation, champion selection, reproduction, the re-evaluations, hall of fame, confirmation, decision sample and files.",
        formula: "Sum over generations of each field of the generation's stage timings (training_evaluation, champion_selection, speciation_and_reproduction, reevaluation_mixed, reevaluation_per_opponent, hall_of_fame, confirmation, decision_sample, checkpoint_and_files). Null for runs from before timings were recorded.",
        reading: "Shows which stage to shrink to speed a run up; with threads the stage times are wall-clock seconds, so they add up to about the run time.",
        caveats: "The event line and the checkpoint are written after the stage timers stop, so a small part of the wall-clock time is in no stage.",
        since_phase: 17,
        json_path: "analysis.timings.{training_evaluation,champion_selection,speciation_and_reproduction,reevaluation_mixed,reevaluation_per_opponent,hall_of_fame,confirmation,decision_sample,checkpoint_and_files}",
    },
    StatInfo {
        id: "analysis_skipped_lines",
        name: "Skipped lines",
        scope: Scope::RunAnalysis,
        unit: "lines",
        range: "0..unbounded",
        meaning: "How many lines of the events file could not be read as events (an unfinished last line, other record types, damage).",
        formula: "The number of non-empty lines of the file that did not parse as a start, generation or run_end event.",
        reading: "0 for a clean file. One skipped line at the end of a file of a stopped run is normal (the line was being written). Many mean the file is damaged or from an incompatible version.",
        caveats: "Skipped lines are not analysed, so a damaged file makes the analysis a statement about the part that could be read.",
        since_phase: 17,
        json_path: "analysis.skipped_lines",
    },
    StatInfo {
        id: "analysis_verdict_still_improving",
        name: "Verdict: still improving",
        scope: Scope::RunAnalysis,
        unit: "verdict",
        range: "present or absent",
        meaning: "The run was still getting better when it ended.",
        formula: "Fires when the peak of the smoothed curve is in the last 5 percent of the run (peak_index >= 0.95 * the last index, more than one generation) or the last-third trend interval lies entirely above 0, and in both cases only if the run did not end below its peak (analysis_verdict_declining does not apply: the last-third interval is not strictly below 0 and stall_fallback is false). While it fires, noise_limited, pool_limited, model_limited, growth_without_gain and stop_early do not. The rules are independent except as described here; several can fire and they are listed in the order of importance. Each message quotes the numbers that made the rule fire.",
        reading: "Train longer (--resume or --from): the plateau has not been reached. A run that stalls at its last generation only because its curve fell is not called still improving.",
        caveats: "A rising last third over a short run can be a recovery from a bad stretch; the intervals of short runs are wide, a fired rule is a hint. A verdict is a heuristic with fixed thresholds, not a statistical test.",
        since_phase: 17,
        json_path: "analysis.verdict.[].{code,message}",
    },
    StatInfo {
        id: "analysis_verdict_declining",
        name: "Verdict: declining",
        scope: Scope::RunAnalysis,
        unit: "verdict",
        range: "present or absent",
        meaning: "The run ended worse than its smoothed peak: the final champion is not the best one.",
        formula: "Fires when the last-third trend interval lies entirely below 0, or the final smoothed value is below peak_smoothed - single_se (stall_fallback is true). The message names the peak generation, the peak and the final smoothed value. The rules are independent except as described here; several can fire and they are listed in the order of importance. Each message quotes the numbers that made the rule fire.",
        reading: "Use the genome of the best generation (the trainer keeps it as best.json) or stop earlier; training longer from this population is unlikely to help. The best generation of a falling run is usually well before its end.",
        caveats: "The final smoothed value covers the last 10 generations, so a recent dip of a few generations does not trigger it. A decline of one noise unit is within what a single evaluation can show: the band is the generous one of the stall rule. A verdict is a heuristic with fixed thresholds, not a statistical test.",
        since_phase: 17,
        json_path: "analysis.verdict.[].{code,message}",
    },
    StatInfo {
        id: "analysis_verdict_noise_limited",
        name: "Verdict: noise limited",
        scope: Scope::RunAnalysis,
        unit: "verdict",
        range: "present or absent",
        meaning: "The last-third trend cannot be told from zero given the noise: what progress remains is not distinguishable from noise.",
        formula: "Fires when still_improving did not fire, the last-third trend interval contains 0, single_se is above 0 and either the change standard deviation is more than 2 times the single-evaluation standard error (analysis_noise_change_sd > 2 * analysis_noise_single_se) or the winner's-curse gap is at least 2 combined standard errors (analysis_curse_reeval_minus_heldout.in_combined_se >= 2). The rules are independent except as described here; several can fire and they are listed in the order of importance. Each message quotes the numbers that made the rule fire.",
        reading: "More --matches-per-genome, the duplicate-deal skill fitness, several seeds and a winner chosen on fresh matches (the Phase 13 recommendation). Without noise_limited a flat curve is attributed to the model (model_limited) or to the pool.",
        caveats: "Both thresholds are fixed heuristics, not a test; a run can be both noisy and limited in other ways. The message says the trend is not distinguishable from zero, not that there is none. A verdict is a heuristic with fixed thresholds, not a statistical test.",
        since_phase: 17,
        json_path: "analysis.verdict.[].{code,message}",
    },
    StatInfo {
        id: "analysis_verdict_pool_limited",
        name: "Verdict: pool limited",
        scope: Scope::RunAnalysis,
        unit: "verdict",
        range: "present or absent",
        meaning: "The opponents no longer teach against some of the pool.",
        formula: "Fires when still_improving did not fire, the last-third trend interval contains 0 and at least one opponent's last-third trend is rising and at least one other opponent's is flat (analysis_opponent_trend classes, which are computed on the last third). The rules are independent except as described here; several can fire and they are listed in the order of importance. Each message quotes the numbers that made the rule fire.",
        reading: "Harder or additional opponents. Flat opponents that are already beaten by a wide margin have nothing left to teach, which is the case this rule is for.",
        caveats: "With few generations most opponent intervals contain 0 and the rule rarely fires; a rising class needs a whole interval above 0. A verdict is a heuristic with fixed thresholds, not a statistical test.",
        since_phase: 17,
        json_path: "analysis.verdict.[].{code,message}",
    },
    StatInfo {
        id: "analysis_verdict_model_limited",
        name: "Verdict: model limited",
        scope: Scope::RunAnalysis,
        unit: "verdict",
        range: "present or absent",
        meaning: "Flat everywhere, above the noise: the model's inputs or the fitness cap what it can learn.",
        formula: "Fires when still_improving did not fire, the last-third trend interval contains 0, no opponent's last-third trend is rising and noise_limited did not fire. The rules are independent except as described here; several can fire and they are listed in the order of importance. Each message quotes the numbers that made the rule fire.",
        reading: "New inputs (--features hidden) or a different fitness. The Phase 13 measurement found that hidden-information inputs beat the base set.",
        caveats: "It is the residual explanation: flat, not noisy and no opponent rising. A poorly measured run lands here too when its noise happens to be just below the noise_limited thresholds. A verdict is a heuristic with fixed thresholds, not a statistical test.",
        since_phase: 17,
        json_path: "analysis.verdict.[].{code,message}",
    },
    StatInfo {
        id: "analysis_verdict_growth_without_gain",
        name: "Verdict: growth without gain",
        scope: Scope::RunAnalysis,
        unit: "verdict",
        range: "present or absent",
        meaning: "The network keeps growing while the score no longer rises.",
        formula: "Fires when still_improving did not fire and analysis_growth.without_gain is true: the mean hidden-node count over the last generations of the last third is above that over its first generations (third_end_nodes > third_start_nodes) and the last-third trend interval contains 0. The rules are independent except as described here; several can fire and they are listed in the order of importance. Each message quotes the numbers that made the rule fire.",
        reading: "A stagnation limit or complexity penalty so the structure stops growing, or a shorter run.",
        caveats: "Compares two means of up to three champions' node counts; a steady one-node gain is enough. A verdict is a heuristic with fixed thresholds, not a statistical test.",
        since_phase: 17,
        json_path: "analysis.verdict.[].{code,message}",
    },
    StatInfo {
        id: "analysis_verdict_converged",
        name: "Verdict: converged",
        scope: Scope::RunAnalysis,
        unit: "verdict",
        range: "present or absent",
        meaning: "The population is nearly uniform: half or more of the genomes sit in the top fitness bin.",
        formula: "Fires when analysis_population_top_bin_share (mean over the last third) is at least 0.5. The rules are independent, several can fire; they are listed in the order of importance. Each message quotes the numbers that made the rule fire.",
        reading: "Raise the mutation rates or the species target to restore variety.",
        caveats: "Depends on the fitness histogram binning; a fitness that saturates at a ceiling also shows it. A verdict is a heuristic with fixed thresholds, not a statistical test.",
        since_phase: 17,
        json_path: "analysis.verdict.[].{code,message}",
    },
    StatInfo {
        id: "analysis_verdict_stop_early",
        name: "Verdict: stop early",
        scope: Scope::RunAnalysis,
        unit: "verdict",
        range: "present or absent",
        meaning: "A long tail of generations added nothing.",
        formula: "Fires when still_improving did not fire, the run has more than one generation, the stall position is at most 0.6 times the last position (stall_index <= 0.6 * (generations - 1)), the last-third trend point estimate is not above 0 and the best generation (from run_end) is not later than the stall generation. The rules are independent except as described here; several can fire and they are listed in the order of importance. Each message quotes the numbers that made the rule fire.",
        reading: "Stop earlier next time; after the stall generation the smoothed score stayed within one noise unit of its peak. The message names the stall generation and how many generations followed.",
        caveats: "The stall rule uses a generous band (one single-evaluation standard error), see analysis_stall_generation; it can name a stall that is too early on a slow climb, which is why a later best generation or a rising point estimate suppresses it. A verdict is a heuristic with fixed thresholds, not a statistical test.",
        since_phase: 17,
        json_path: "analysis.verdict.[].{code,message}",
    },
    StatInfo {
        id: "analysis_verdict_no_clear_limit",
        name: "Verdict: no clear limit",
        scope: Scope::RunAnalysis,
        unit: "verdict",
        range: "present or absent",
        meaning: "No rule fired.",
        formula: "Appears alone when none of the other eight rules fires. The rules are independent except as described here; several can fire and they are listed in the order of importance. Each message quotes the numbers that made the rule fire.",
        reading: "Read the numbers and charts yourself; the absence of a finding is not a finding of health.",
        caveats: "Every rule has its thresholds; a run just inside all of them gets this verdict. A verdict is a heuristic with fixed thresholds, not a statistical test.",
        since_phase: 17,
        json_path: "analysis.verdict.[].{code,message}",
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

    const REQUIRED_IDS: [&str; 24] = [
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
        "voluntary_pass_rate_intervals",
        "plain_mean_role_score",
        "variance_explained",
        "feature_r2",
        "estimator_features",
        "difference_in_se",
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
            assert_ne!(scope.title(), "");
        }
        for s in catalog() {
            assert!(Scope::ALL.contains(&s.scope));
        }
    }

    #[test]
    fn scope_serialises_in_snake_case() {
        let json = serde_json::to_string(&Scope::PerStrategy).unwrap();
        assert_eq!(json, "\"per_strategy\"");
        let json = serde_json::to_string(&Scope::RunAnalysis).unwrap();
        assert_eq!(json, "\"run_analysis\"");
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
        assert_eq!(render_explain(&["bogus"]), "");
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
