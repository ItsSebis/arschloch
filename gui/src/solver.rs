//! The solver position editor's model: a 52-card grid, the table and the other
//! seats' card counts, turned into the position text of `docs/SOLVER.md`, and
//! the ranked-moves table read back from `cli solve`. No drawing code, so it is
//! unit-tested. The window runs the real `cli solve`; nothing is solved here.

use std::fmt::Write as _;

pub const RANKS: [&str; 13] = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A",
];
pub const SUITS: [char; 4] = ['C', 'D', 'H', 'S'];

/// Where a card is, from the point of view of the seat that moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// Already played this round (the default: an endgame has few cards left).
    Played,
    /// In our hand.
    Hand,
    /// In the combo on the table.
    Table,
    /// Held by another seat; the solver considers every split.
    Unseen,
}

impl Place {
    pub const ALL: [Place; 4] = [Place::Hand, Place::Table, Place::Unseen, Place::Played];

    pub fn label(self) -> &'static str {
        match self {
            Place::Played => "played",
            Place::Hand => "my hand",
            Place::Table => "on the table",
            Place::Unseen => "unseen",
        }
    }
}

/// `9H`, `10S`, `AC`: the card at grid index `rank * 4 + suit`.
pub fn card_token(index: usize) -> String {
    format!("{}{}", RANKS[index / 4], SUITS[index % 4])
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolverForm {
    pub players: u8,
    /// The seat that moves.
    pub seat: u8,
    pub cards: [Place; 52],
    /// Cards left per seat; the entry of our own seat is ignored (the hand decides).
    pub sizes: Vec<usize>,
    /// The seat whose combo is on the table; `None` when we lead a fresh trick.
    pub holder: Option<u8>,
    /// Seats the user marked as passed; the seats between the holder and us are added.
    pub passed: Vec<bool>,
    pub max_unseen: usize,
    pub budget: u64,
}

impl Default for SolverForm {
    fn default() -> Self {
        Self {
            players: 4,
            seat: 0,
            cards: [Place::Played; 52],
            sizes: vec![0; 4],
            holder: None,
            passed: vec![false; 4],
            max_unseen: 12,
            budget: 50_000_000,
        }
    }
}

impl SolverForm {
    /// The endgame of `docs/examples/solve-endgame.txt`: we lead with `5C 9D KH`,
    /// the others hold two cards each, six cards are unseen.
    pub fn example() -> Self {
        let mut form = Self {
            sizes: vec![3, 2, 2, 2],
            ..Self::default()
        };
        for token in ["5C", "9D", "KH"] {
            form.set(token, Place::Hand);
        }
        // The deck minus the played cards of the example file are the unseen ones.
        for token in ["3D", "4H", "6S", "7C", "10D", "JH"] {
            form.set(token, Place::Unseen);
        }
        form
    }

    fn set(&mut self, token: &str, place: Place) {
        if let Some(i) = (0..52).find(|&i| card_token(i) == token) {
            self.cards[i] = place;
        }
    }

    pub fn set_players(&mut self, players: u8) {
        self.players = players.clamp(3, 6);
        let n = usize::from(self.players);
        self.sizes.resize(n, 0);
        self.passed.resize(n, false);
        self.seat = self.seat.min(self.players - 1);
        if self
            .holder
            .is_some_and(|h| h >= self.players || h == self.seat)
        {
            self.holder = None;
        }
    }

    pub fn fill(&mut self, place: Place) {
        self.cards = [place; 52];
    }

    pub fn count(&self, place: Place) -> usize {
        self.cards.iter().filter(|&&c| c == place).count()
    }

    fn tokens(&self, place: Place) -> Vec<String> {
        (0..52)
            .filter(|&i| self.cards[i] == place)
            .map(card_token)
            .collect()
    }

    /// Cards left per seat as the position file states them.
    pub fn seat_sizes(&self) -> Vec<usize> {
        (0..usize::from(self.players))
            .map(|s| {
                if s == usize::from(self.seat) {
                    self.count(Place::Hand)
                } else {
                    self.sizes.get(s).copied().unwrap_or(0)
                }
            })
            .collect()
    }

    /// The seats that passed: the marked ones plus every seat with cards between
    /// the holder and us (they acted after the holder, so they passed or are out).
    pub fn passed_seats(&self) -> Vec<usize> {
        let Some(holder) = self.holder else {
            return Vec::new();
        };
        let n = usize::from(self.players);
        let sizes = self.seat_sizes();
        let between: Vec<usize> = (1..n)
            .map(|k| (usize::from(holder) + k) % n)
            .take_while(|&s| s != usize::from(self.seat))
            .collect();
        (0..n)
            .filter(|&s| {
                s != usize::from(self.seat)
                    && sizes[s] > 0
                    && (self.passed.get(s).copied().unwrap_or(false) || between.contains(&s))
            })
            .collect()
    }

    /// What `cli solve` would reject, in words; `None` when the position can be written.
    pub fn problem(&self) -> Option<String> {
        let hand = self.count(Place::Hand);
        if hand == 0 {
            return Some("paint at least one card as `my hand`".to_owned());
        }
        let table = self.tokens(Place::Table);
        match (self.holder, table.is_empty()) {
            (None, false) => {
                return Some(
                    "cards are on the table but no seat holds them: pick the holder, or paint them back"
                        .to_owned(),
                )
            }
            (Some(_), true) => {
                return Some("a seat holds the table but no card is painted `on the table`".to_owned())
            }
            _ => {}
        }
        if let Some(holder) = self.holder {
            if holder == self.seat {
                return Some("the table's holder is another seat than ours".to_owned());
            }
            let ranks: Vec<usize> = (0..52)
                .filter(|&i| self.cards[i] == Place::Table)
                .map(|i| i / 4)
                .collect();
            if ranks.iter().any(|&r| r != ranks[0]) {
                return Some("the cards on the table must all have the same rank".to_owned());
            }
        }
        let others: usize = self
            .seat_sizes()
            .iter()
            .enumerate()
            .filter(|&(s, _)| s != usize::from(self.seat))
            .map(|(_, n)| n)
            .sum();
        let unseen = self.count(Place::Unseen);
        if unseen != others {
            return Some(format!(
                "{unseen} cards are painted unseen but the other seats hold {others}: fix the counts or the cards"
            ));
        }
        if unseen == 0 {
            return Some("no unseen cards: nothing to analyse".to_owned());
        }
        if unseen > self.max_unseen {
            return Some(format!(
                "{unseen} unseen cards are more than `max unseen` ({}): the search would be too big",
                self.max_unseen
            ));
        }
        None
    }

    /// The position file text (`docs/SOLVER.md`).
    pub fn text(&self) -> String {
        let sizes: Vec<String> = self.seat_sizes().iter().map(ToString::to_string).collect();
        let mut out = format!(
            "players: {}\ndeck: single\nseat: {}\nhand: {}\nsizes: {}\n",
            self.players,
            self.seat,
            self.tokens(Place::Hand).join(" "),
            sizes.join(" ")
        );
        match self.holder {
            Some(holder) => {
                let _ = writeln!(
                    out,
                    "table: {} by {holder}",
                    self.tokens(Place::Table).join(" ")
                );
                let passed: Vec<String> = self
                    .passed_seats()
                    .iter()
                    .map(ToString::to_string)
                    .collect();
                let _ = writeln!(out, "passed: {}", passed.join(" "));
            }
            None => out.push_str("table: lead\n"),
        }
        let _ = writeln!(out, "played: {}", self.tokens(Place::Played).join(" "));
        out
    }

    /// The arguments of `cli solve` for the position file at `path`.
    pub fn args(&self, path: &str) -> Vec<String> {
        vec![
            "solve".to_owned(),
            path.to_owned(),
            "--max-unseen".to_owned(),
            self.max_unseen.to_string(),
            "--budget".to_owned(),
            self.budget.to_string(),
        ]
    }
}

/// One ranked move of `cli solve`'s table.
#[derive(Clone, Debug, PartialEq)]
pub struct MoveRow {
    pub text: String,
    /// Expected finishing place, 1 = first out.
    pub place: f64,
    pub score: f64,
    pub best_worst: String,
    pub best_share: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// The lines above the table (who moves, how many deals).
    pub header: Vec<String>,
    /// Best move first.
    pub rows: Vec<MoveRow>,
    /// Lines below the table (a note about identical cards).
    pub notes: Vec<String>,
}

/// Reads the output of `cli solve`; the error is what the command said instead.
///
/// # Errors
///
/// The text has no moves table (the command failed): its own words.
pub fn parse_report(text: &str) -> Result<Report, String> {
    let mut header = Vec::new();
    let mut rows = Vec::new();
    let mut notes = Vec::new();
    let mut in_table = false;
    for line in text.lines().map(str::trim_end).filter(|l| !l.is_empty()) {
        if line.starts_with("[finished") || line.starts_with("[cancelled") {
            continue;
        }
        if line.starts_with("move ") && line.contains("exp.place") {
            in_table = true;
        } else if !in_table {
            header.push(line.to_owned());
        } else if let Some(row) = parse_row(line) {
            rows.push(row);
        } else {
            notes.push(line.to_owned());
        }
    }
    if rows.is_empty() {
        let said = header.join(" ");
        return Err(if said.is_empty() {
            "the solver printed no moves".to_owned()
        } else {
            said
        });
    }
    Ok(Report {
        header,
        rows,
        notes,
    })
}

/// `pair 8H 8S      1.50    +0.33    1/2   75%`: the last four columns are numbers.
fn parse_row(line: &str) -> Option<MoveRow> {
    let words: Vec<&str> = line.split_whitespace().collect();
    let n = words.len();
    if n < 5 {
        return None;
    }
    let place = words[n - 4].parse().ok()?;
    let score = words[n - 3].parse().ok()?;
    if !words[n - 2].contains('/') || !words[n - 1].ends_with('%') {
        return None;
    }
    Some(MoveRow {
        text: words[..n - 4].join(" "),
        place,
        score,
        best_worst: words[n - 2].to_owned(),
        best_share: words[n - 1].to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{real_cli, TempDir};

    const EXAMPLE: &str = "move                   exp.place    score best/worst   best in\n";

    #[test]
    fn card_tokens_cover_the_deck() {
        assert_eq!(card_token(0), "2C");
        assert_eq!(card_token(51), "AS");
        assert_eq!(card_token(8 * 4 + 3), "10S");
        let all: std::collections::HashSet<String> = (0..52).map(card_token).collect();
        assert_eq!(all.len(), 52);
    }

    #[test]
    fn the_example_is_a_valid_position() {
        let form = SolverForm::example();
        assert_eq!(form.problem(), None, "{:?}", form.problem());
        assert_eq!(form.count(Place::Hand), 3);
        assert_eq!(form.count(Place::Unseen), 6);
        assert_eq!(form.count(Place::Played), 43);
        let text = form.text();
        assert!(text.starts_with(
            "players: 4\ndeck: single\nseat: 0\nhand: 5C 9D KH\nsizes: 3 2 2 2\ntable: lead\n"
        ));
        assert!(!text.contains("passed"), "no pass line when we lead");
    }

    #[test]
    fn the_text_of_the_example_describes_the_same_position_as_the_shipped_file() {
        // Same hand, sizes and table; the played cards are the deck minus hand and unseen.
        let shipped = include_str!("../../docs/examples/solve-endgame.txt");
        let played_of = |text: &str| {
            let line = text
                .lines()
                .find(|l| l.starts_with("played:"))
                .unwrap()
                .trim_start_matches("played:");
            let mut v: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
            v.sort();
            v
        };
        let mine = SolverForm::example().text();
        assert_eq!(played_of(&mine), played_of(shipped));
    }

    #[test]
    fn a_table_with_a_holder_writes_the_pass_line() {
        let mut form = SolverForm {
            sizes: vec![0, 2, 1, 2],
            ..SolverForm::default()
        };
        for t in ["9C", "KD"] {
            form.set(t, Place::Hand);
        }
        for t in ["8H", "8S"] {
            form.set(t, Place::Table);
        }
        for t in ["2D", "3D", "4D", "5D", "6D"] {
            form.set(t, Place::Unseen);
        }
        form.holder = Some(2);
        // Seat 3 acts between the holder (2) and us (0), so it passed without being told.
        assert_eq!(form.passed_seats(), [3]);
        assert_eq!(form.problem(), None);
        let text = form.text();
        assert!(
            text.contains("sizes: 2 2 1 2\ntable: 8H 8S by 2\npassed: 3\n"),
            "{text}"
        );
        form.passed[1] = true;
        assert_eq!(form.passed_seats(), [1, 3]);
    }

    #[test]
    fn mistakes_are_named() {
        let mut form = SolverForm::example();
        form.sizes[1] = 3;
        assert!(form.problem().unwrap().contains("unseen"));
        let mut form = SolverForm::example();
        form.fill(Place::Played);
        assert!(form.problem().unwrap().contains("my hand"));
        let mut form = SolverForm::example();
        form.set("8H", Place::Table);
        assert!(form.problem().unwrap().contains("holds"));
        let mut form = SolverForm::example();
        form.holder = Some(1);
        assert!(form.problem().unwrap().contains("no card"));
        let mut form = SolverForm::example();
        form.set("8H", Place::Table);
        form.set("9S", Place::Table);
        form.holder = Some(1);
        form.sizes = vec![3, 2, 2, 2];
        assert!(form.problem().unwrap().contains("same rank"));
        let mut form = SolverForm::example();
        form.max_unseen = 4;
        assert!(form.problem().unwrap().contains("max unseen"));
        let mut form = SolverForm::example();
        form.fill(Place::Played);
        form.set("AS", Place::Hand);
        form.sizes = vec![1, 0, 0, 0];
        assert!(form.problem().unwrap().contains("no unseen"));
    }

    #[test]
    fn changing_the_table_size_keeps_everything_in_range() {
        let mut form = SolverForm::example();
        form.seat = 3;
        form.holder = Some(2);
        form.set_players(3);
        assert_eq!((form.players, form.seat, form.holder), (3, 2, None));
        assert_eq!(form.sizes.len(), 3);
        form.set_players(9);
        assert_eq!(form.players, 6);
        assert_eq!(form.passed.len(), 6);
    }

    #[test]
    fn the_ranked_table_is_read_back() {
        let text = format!(
            "seat 0 to move, hand: 5C 9D KH\ndeals considered: 90 (351 search nodes)\n{EXAMPLE}\
             pair 8H 8S                1.50    +0.33    1/2    75%\n\
             pass                      2.10    -0.20    1/4     5%\n\
             note: identical cards of a double deck are told apart\n[finished: exit code 0]\n"
        );
        let report = parse_report(&text).unwrap();
        assert_eq!(report.header.len(), 2);
        assert_eq!(report.rows.len(), 2);
        assert_eq!(report.rows[0].text, "pair 8H 8S");
        assert!((report.rows[0].place - 1.5).abs() < 1e-9);
        assert!((report.rows[0].score - 0.33).abs() < 1e-9);
        assert_eq!(report.rows[0].best_worst, "1/2");
        assert_eq!(report.rows[0].best_share, "75%");
        assert_eq!(report.rows[1].text, "pass");
        assert_eq!(report.notes.len(), 1);
    }

    #[test]
    fn a_failed_command_gives_its_own_words() {
        let text = "Error: x.txt: sizes: expected 4 numbers\n[finished: exit code 1]\n";
        assert_eq!(
            parse_report(text).unwrap_err(),
            "Error: x.txt: sizes: expected 4 numbers"
        );
        assert!(parse_report("").is_err());
    }

    /// The real `cli solve` accepts the editor's text and its table parses.
    #[test]
    fn the_real_cli_solves_the_editors_position_and_the_table_parses() {
        let Some(cli) = real_cli() else { return };
        let dir = TempDir::new("solver");
        let file = dir.0.join("position.txt");
        let form = SolverForm::example();
        std::fs::write(&file, form.text()).unwrap();
        let out = std::process::Command::new(cli)
            .args(form.args(&file.to_string_lossy()))
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{text}");
        let report = parse_report(&text).unwrap();
        assert!(report.rows.len() >= 3, "{text}");
        assert!(report.header[1].contains("deals considered: 90"), "{text}");
        // Ranked best first.
        assert!(
            report.rows.windows(2).all(|w| w[0].place <= w[1].place),
            "{text}"
        );
        // A trick in progress (a table and a pass) is accepted too.
        let mut trick = SolverForm {
            sizes: vec![0, 2, 1, 2],
            holder: Some(2),
            ..SolverForm::default()
        };
        for t in ["9C", "KD"] {
            trick.set(t, Place::Hand);
        }
        for t in ["8H", "8S"] {
            trick.set(t, Place::Table);
        }
        for t in ["2D", "3D", "4D", "5D", "6D"] {
            trick.set(t, Place::Unseen);
        }
        assert_eq!(trick.problem(), None);
        std::fs::write(&file, trick.text()).unwrap();
        let out = std::process::Command::new(real_cli().unwrap())
            .args(trick.args(&file.to_string_lossy()))
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{text}");
        let rows = parse_report(&text).unwrap().rows;
        assert!(rows.iter().any(|r| r.text == "pass"), "{text}");
        // A wrong count is reported by the command, and the editor says the same first.
        let mut bad = form.clone();
        bad.sizes[1] = 3;
        assert!(bad.problem().is_some());
        std::fs::write(&file, bad.text()).unwrap();
        let out = std::process::Command::new(real_cli().unwrap())
            .args(bad.args(&file.to_string_lossy()))
            .output()
            .unwrap();
        assert!(!out.status.success(), "the negative control must fail");
    }
}
