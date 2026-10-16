//! `cli solve FILE`: the exact analysis of one real situation
//! (docs/SOLVER.md). The position file format is described there.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Context;
use engine::{standard_deck, Card, Combo, DeckVariant, DuplicateRule, SeatId};
use sim::hand_reading::PassCeilings;
use sim::solver::notation::{card_token, parse_card_token};
use sim::solver::{analyze, MoveReport, MoveStats, Observation, SolverConfig};

const USAGE: &str = "usage: cli solve FILE [--max-unseen N] [--budget N]   (see docs/SOLVER.md)";

pub fn run(args: impl Iterator<Item = String>) -> anyhow::Result<()> {
    let mut path: Option<PathBuf> = None;
    let mut config = SolverConfig {
        max_unseen: 12,
        node_budget: 50_000_000,
    };
    let mut args = args;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max-unseen" => {
                config.max_unseen = args
                    .next()
                    .context("--max-unseen needs a number")?
                    .parse()
                    .context("--max-unseen needs a number")?;
            }
            "--budget" => {
                config.node_budget = args
                    .next()
                    .context("--budget needs a number")?
                    .parse()
                    .context("--budget needs a number")?;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other if path.is_none() && !other.starts_with("--") => {
                path = Some(PathBuf::from(other));
            }
            other => anyhow::bail!("unexpected argument `{other}`\n{USAGE}"),
        }
    }
    let path = path.context(USAGE)?;
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let observation =
        parse_position(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let report = analyze(&observation, &config).map_err(|e| {
        anyhow::anyhow!("cannot solve this position: {e:?} (try --max-unseen / --budget)")
    })?;
    print!("{}", render(&report, &observation));
    Ok(())
}

/// The words that may precede the cards of the combo on the table.
const SIZE_WORDS: [&str; 8] = [
    "single", "pair", "triple", "quad", "five", "six", "seven", "eight",
];

fn seat_list(text: &str, key: &str) -> Result<Vec<usize>, String> {
    text.split_whitespace()
        .map(|word| {
            word.parse::<usize>()
                .map_err(|_| format!("{key}: `{word}` is not a seat number"))
        })
        .collect()
}

/// `key: value` lines, `#` starts a comment, keys are case-insensitive.
fn read_values(text: &str) -> Result<HashMap<String, String>, String> {
    let mut values: HashMap<String, String> = HashMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once(':')
            .ok_or_else(|| format!("`{line}` is not `key: value`"))?;
        let key = key.trim().to_ascii_lowercase();
        if values
            .insert(key.clone(), value.trim().to_owned())
            .is_some()
        {
            return Err(format!("`{key}` is given twice"));
        }
    }
    Ok(values)
}

/// `lead`, or the combo on the table and who holds it: `pair 8H 8S by 2`.
fn parse_table(
    description: &str,
    n: usize,
    take: &mut impl FnMut(&str) -> Result<Card, String>,
) -> Result<Option<(SeatId, Combo)>, String> {
    if description == "lead" {
        return Ok(None);
    }
    let words: Vec<&str> = description.split_whitespace().collect();
    let by = words.iter().position(|w| *w == "by").ok_or(
        "table: write `lead` or e.g. `pair 8H 8S by 2` (cards, then `by` and the seat holding them)",
    )?;
    let holder: SeatId = words
        .get(by + 1)
        .and_then(|w| w.parse().ok())
        .filter(|s| usize::from(*s) < n)
        .ok_or("table: `by` must be followed by a seat number")?;
    let mut cards = Vec::new();
    for word in &words[..by] {
        if SIZE_WORDS.contains(word) {
            continue;
        }
        cards.push(take(word)?);
    }
    let combo = Combo::from_slice(&cards).ok_or("table: the cards must all have the same rank")?;
    Ok(Some((holder, combo)))
}

/// The seats that passed in this trick: seats other than ours that are still
/// in the round, and only when a combo is on the table.
fn parse_passed(
    text: &str,
    seat: SeatId,
    sizes: &[usize],
    combo_on_table: bool,
) -> Result<Vec<usize>, String> {
    let passed = seat_list(text, "passed")?;
    if let Some(bad) = passed
        .iter()
        .find(|&&s| s >= sizes.len() || s == usize::from(seat) || sizes[s] == 0)
    {
        return Err(format!(
            "passed: seat {bad} cannot have passed (not a seat, our own seat, or already out)"
        ));
    }
    if !combo_on_table && !passed.is_empty() {
        return Err("passed: nobody can have passed when we lead a fresh trick".to_owned());
    }
    Ok(passed)
}

/// Reads a position file (see docs/SOLVER.md) into what the solver needs.
pub(crate) fn parse_position(text: &str) -> Result<Observation, String> {
    let values = read_values(text)?;
    let get = |key: &str| values.get(key).map(String::as_str);
    let need = |key: &str| get(key).ok_or_else(|| format!("missing `{key}`"));

    let players: u8 = need("players")?
        .parse()
        .ok()
        .filter(|p| (3..=6).contains(p))
        .ok_or("players: must be a number from 3 to 6")?;
    let n = usize::from(players);
    let variant = match get("deck").unwrap_or("single") {
        "single" => DeckVariant::Single,
        "double" => DeckVariant::Double,
        other => return Err(format!("deck: `{other}` is not single or double")),
    };
    let duplicate_rule = match get("duplicate").unwrap_or("first") {
        "first" => DuplicateRule::FirstDealtWins,
        "last" => DuplicateRule::LastDealtWins,
        other => return Err(format!("duplicate: `{other}` is not first or last")),
    };
    let seat: SeatId = need("seat")?
        .parse()
        .ok()
        .filter(|s| usize::from(*s) < n)
        .ok_or("seat: must be a seat number below `players`")?;
    let hand_text = need("hand")?.to_owned();
    let sizes: Vec<usize> = need("sizes")?
        .split_whitespace()
        .map(|word| {
            word.parse::<usize>()
                .map_err(|_| format!("sizes: `{word}` is not a number"))
        })
        .collect::<Result<_, _>>()?;
    if sizes.len() != n {
        return Err(format!(
            "sizes: expected {n} numbers (one per seat), got {}",
            sizes.len()
        ));
    }

    // The deck, numbered; every listed card consumes one copy of it.
    let mut pool: Vec<Card> = standard_deck(variant);
    for (index, card) in pool.iter_mut().enumerate() {
        card.deal_index = u8::try_from(index).map_err(|_| "the deck has too many cards")?;
    }
    let mut take = |token: &str| -> Result<Card, String> {
        let (rank, suit) = parse_card_token(token)
            .ok_or_else(|| format!("`{token}` is not a card (try 9H, 10S, AC)"))?;
        let position = pool
            .iter()
            .position(|c| c.rank == rank && c.suit == suit)
            .ok_or_else(|| format!("no more copies of {token} in the deck"))?;
        Ok(pool.remove(position))
    };

    let hand: Vec<Card> = hand_text
        .split_whitespace()
        .map(&mut take)
        .collect::<Result<_, _>>()?;
    if sizes[usize::from(seat)] != hand.len() {
        return Err(format!(
            "sizes: seat {seat} holds {} cards but the hand lists {}",
            sizes[usize::from(seat)],
            hand.len()
        ));
    }
    for token in get("played").unwrap_or("").split_whitespace() {
        take(token)?;
    }

    let on_table = parse_table(get("table").unwrap_or("lead"), n, &mut take)?;

    let passed = parse_passed(
        get("passed").unwrap_or(""),
        seat,
        &sizes,
        on_table.is_some(),
    )?;

    let others: usize = (0..n)
        .filter(|&s| s != usize::from(seat))
        .map(|s| sizes[s])
        .sum();
    if pool.len() != others {
        return Err(format!(
            "the deck has {} unseen cards (not in the hand, `played` or on the table) but the other seats hold {others} (sizes)",
            pool.len()
        ));
    }

    let in_trick: Vec<bool> = (0..n)
        .map(|s| sizes[s] > 0 && !passed.contains(&s))
        .collect();
    Ok(Observation {
        duplicate_rule,
        seat,
        player_count: players,
        hand,
        hand_sizes: sizes,
        in_trick,
        ceilings: vec![PassCeilings::default(); n],
        unseen: pool,
        on_table,
    })
}

fn move_name(stats: &MoveStats) -> String {
    match stats.mv {
        engine::Move::Pass => "pass".to_owned(),
        engine::Move::Play(combo) => combo
            .cards()
            .iter()
            .map(card_token)
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// The ranked moves as text; `exp.place` is 1-based (1 = first out).
pub(crate) fn render(report: &MoveReport, observation: &Observation) -> String {
    use std::fmt::Write as _;
    let mut order: Vec<&MoveStats> = report.moves.iter().collect();
    order.sort_by(|a, b| a.mean_place.total_cmp(&b.mean_place));
    let mut out = String::new();
    let _ = writeln!(
        out,
        "seat {} to move, hand: {}",
        observation.seat,
        observation
            .hand
            .iter()
            .map(card_token)
            .collect::<Vec<_>>()
            .join(" ")
    );
    let _ = writeln!(
        out,
        "deals considered: {} ({} search nodes)",
        report.deals, report.nodes
    );
    let _ = writeln!(
        out,
        "{:<22} {:>9} {:>8} {:>10} {:>9}",
        "move", "exp.place", "score", "best/worst", "best in"
    );
    for stats in order {
        let _ = writeln!(
            out,
            "{:<22} {:>9.2} {:>+8.2} {:>6}/{:<3} {:>8.0}%",
            move_name(stats),
            stats.mean_place + 1.0,
            report.score(stats),
            stats.best_place + 1,
            stats.worst_place + 1,
            stats.best_share * 100.0
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{standard_deck, DeckVariant};
    use sim::solver::notation::card_token;

    /// Three players, we hold `AC` and lead; the two other cards left in the
    /// deck are `2D` and `3D`, one for each opponent.
    fn endgame_text() -> String {
        let played: Vec<String> = standard_deck(DeckVariant::Single)
            .iter()
            .map(card_token)
            .filter(|t| !["AC", "2D", "3D"].contains(&t.as_str()))
            .collect();
        format!(
            "players: 3\ndeck: single\nseat: 0\nhand: AC\nsizes: 1 1 1\ntable: lead\nplayed: {}\n",
            played.join(" ")
        )
    }

    #[test]
    fn a_position_file_parses_and_solves() {
        let observation = parse_position(&endgame_text()).unwrap();
        assert_eq!(observation.unseen.len(), 2);
        let report =
            sim::solver::analyze(&observation, &sim::solver::SolverConfig::default()).unwrap();
        let text = render(&report, &observation);
        assert!(text.contains("AC"), "{text}");
        assert!(text.contains("deals considered: 2"), "{text}");
    }

    #[test]
    fn comments_blank_lines_and_key_case_are_ignored() {
        let text = format!(
            "# a position\n\n{}",
            endgame_text().replace("players: 3", "Players: 3   # three seats")
        );
        let parsed = parse_position(&text);
        assert!(parsed.is_ok(), "{:?}", parsed.err());
    }

    #[test]
    fn mistakes_are_reported_in_plain_words() {
        assert!(parse_position("players: 4\n").unwrap_err().contains("seat"));
        let missing_hand = endgame_text().replace("hand: AC\n", "");
        assert!(parse_position(&missing_hand).unwrap_err().contains("hand"));
        let bad_size = endgame_text().replace("sizes: 1 1 1", "sizes: 2 1 1");
        assert!(parse_position(&bad_size).unwrap_err().contains("sizes"));
        let too_many = endgame_text().replace("hand: AC", "hand: AC AC");
        assert!(parse_position(&too_many).unwrap_err().contains("AC"));
        let wrong_unseen = endgame_text().replace("sizes: 1 1 1", "sizes: 1 1 2");
        assert!(parse_position(&wrong_unseen)
            .unwrap_err()
            .contains("unseen"));
        let not_a_card = endgame_text().replace("hand: AC", "hand: 1X");
        assert!(parse_position(&not_a_card).unwrap_err().contains("1X"));
        let bad_players = endgame_text().replace("players: 3", "players: 9");
        assert!(parse_position(&bad_players)
            .unwrap_err()
            .contains("players"));
    }

    #[test]
    fn a_trick_in_progress_with_a_pass_parses() {
        // Four players; seat 2 holds the combo on the table (a pair of 8s), seat 1 passed.
        let all: Vec<String> = standard_deck(DeckVariant::Single)
            .iter()
            .map(card_token)
            .collect();
        let mine = ["9C", "KD"];
        let table = ["8H", "8S"];
        let unseen = ["2D", "3D", "4D", "5D", "6D"]; // seats 1 (2 cards), 2 (1), 3 (2)
        let played: Vec<&String> = all
            .iter()
            .filter(|t| !mine.contains(&t.as_str()))
            .filter(|t| !table.contains(&t.as_str()))
            .filter(|t| !unseen.contains(&t.as_str()))
            .collect();
        let text = format!(
            "players: 4\nseat: 0\nhand: {}\nsizes: 2 2 1 2\ntable: pair {} by 2\npassed: 1\nplayed: {}\n",
            mine.join(" "),
            table.join(" "),
            played.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ")
        );
        let observation = parse_position(&text).unwrap();
        assert_eq!(observation.on_table.map(|(holder, _)| holder), Some(2));
        assert_eq!(observation.in_trick, vec![true, false, true, true]);
        assert_eq!(observation.unseen.len(), 5);
    }
}
