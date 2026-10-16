//! The window's drawing helpers: the solver's seat table and card grid, the
//! recommendation editor, the strategy rows and small widgets. They only draw
//! and edit the forms of `form.rs`, `solver.rs` and `advisor.rs`; `main.rs`
//! owns the app state and the jobs.

use eframe::egui;

use crate::advisor::Recommendation;
use crate::form::Form;
use crate::solver::{Place, Report, SolverForm};
use crate::PickTarget;
use crate::STRATEGIES;

/// The other seats' card counts, who passed and who holds the table, and the search limits.
pub fn seat_rows(ui: &mut egui::Ui, form: &mut SolverForm) {
    egui::Grid::new("solver-seats").show(ui, |ui| {
        ui.label("Seat");
        ui.label("Cards left");
        ui.label("Passed");
        ui.end_row();
        let derived = form.passed_seats();
        let hand = form.count(Place::Hand);
        for seat in 0..usize::from(form.players) {
            let mine = seat == usize::from(form.seat);
            ui.label(if mine {
                format!("{seat} (me)")
            } else {
                seat.to_string()
            });
            if mine {
                ui.label(format!("{hand} (my hand)"));
                ui.label("");
            } else {
                ui.add(egui::DragValue::new(&mut form.sizes[seat]).range(0..=26));
                let marked = form.passed[seat];
                let mut shown = marked || derived.contains(&seat);
                // The holder of the table never passed: no tick box, and no stale mark.
                if form.holder == u8::try_from(seat).ok() {
                    shown = false;
                    ui.add_enabled(false, egui::Checkbox::without_text(&mut shown));
                } else {
                    let editable = form.holder.is_some() && !derived.contains(&seat) || marked;
                    if ui
                        .add_enabled(editable, egui::Checkbox::without_text(&mut shown))
                        .changed()
                    {
                        form.passed[seat] = shown;
                    }
                }
            }
            ui.end_row();
        }
    });
    ui.horizontal(|ui| {
        ui.label("Combo on the table is held by");
        let text = form
            .holder
            .map_or_else(|| "nobody (I lead)".to_owned(), |h| format!("seat {h}"));
        egui::ComboBox::from_id_salt("holder")
            .selected_text(text)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut form.holder, None, "nobody (I lead)");
                for seat in 0..form.players {
                    if seat != form.seat {
                        ui.selectable_value(&mut form.holder, Some(seat), format!("seat {seat}"));
                    }
                }
            });
        ui.add(
            egui::DragValue::new(&mut form.max_unseen)
                .range(1..=20)
                .prefix("max unseen "),
        );
        ui.add(
            egui::DragValue::new(&mut form.budget)
                .range(1_000..=u64::MAX)
                .speed(100_000.0)
                .prefix("node budget "),
        );
    });
}

/// The recommended training as editable fields; `start` gets the arguments when its button is pressed.
pub fn recommendation_editor(
    ui: &mut egui::Ui,
    rec: &mut Recommendation,
    idle: bool,
    pick: &mut Option<PickTarget>,
    start: &mut Option<Vec<String>>,
) {
    ui.separator();
    ui.strong("Recommended training (editable)");
    egui::Grid::new("recommendation")
        .num_columns(4)
        .show(ui, |ui| {
            number(ui, "Population", &mut rec.population);
            number(ui, "Generations", &mut rec.generations);
            ui.end_row();
            number(ui, "Matches per genome", &mut rec.matches_per_genome);
            number(ui, "Rounds", &mut rec.rounds);
            ui.end_row();
            number(ui, "Hall of fame (0 = off)", &mut rec.hall_of_fame);
            number(ui, "Hall interval", &mut rec.hall_interval);
            ui.end_row();
            ui.label("Run folder");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut rec.out);
                if ui.button("Browse").clicked() {
                    *pick = Some(PickTarget::TrainOut);
                }
            });
            ui.label("Opponents");
            ui.label(rec.opponents.join(" "));
            ui.end_row();
        });
    let problem = rec.problem();
    if let Some(problem) = &problem {
        ui.colored_label(egui::Color32::LIGHT_RED, problem);
    }
    ui.monospace(format!("cli {}", rec.args().join(" ")));
    if ui
        .add_enabled(
            problem.is_none() && idle,
            egui::Button::new("Start this training"),
        )
        .clicked()
    {
        *start = Some(rec.args());
    }
}

/// The 52 cards as buttons, a row per suit; a click paints the card with the brush.
pub fn card_grid(ui: &mut egui::Ui, form: &mut SolverForm, brush: Place) {
    egui::Grid::new("cards").spacing([3.0, 3.0]).show(ui, |ui| {
        for suit in 0..4 {
            for rank in 0..13 {
                let index = rank * 4 + suit;
                let place = form.cards[index];
                let fill = match place {
                    Place::Hand => egui::Color32::from_rgb(40, 130, 60),
                    Place::Table => egui::Color32::from_rgb(190, 120, 30),
                    Place::Unseen => egui::Color32::from_rgb(50, 90, 170),
                    Place::Played => egui::Color32::from_gray(50),
                };
                let button = egui::Button::new(
                    egui::RichText::new(crate::solver::card_token(index))
                        .color(egui::Color32::WHITE),
                )
                .fill(fill)
                .min_size(egui::vec2(36.0, 24.0));
                if ui.add(button).on_hover_text(place.label()).clicked() {
                    form.cards[index] = brush;
                }
            }
            ui.end_row();
        }
    });
}

/// The ranked moves of `cli solve`, best first.
pub fn solver_report(ui: &mut egui::Ui, report: &Report) {
    for line in &report.header {
        ui.label(line);
    }
    egui::Grid::new("moves").striped(true).show(ui, |ui| {
        for title in ["Move", "Exp. place", "Score", "Best/worst", "Best in"] {
            ui.strong(title);
        }
        ui.end_row();
        for (rank, row) in report.rows.iter().enumerate() {
            let text = egui::RichText::new(&row.text);
            ui.label(if rank == 0 { text.strong() } else { text });
            ui.label(format!("{:.2}", row.place));
            ui.label(format!("{:+.2}", row.score));
            ui.label(&row.best_worst);
            ui.label(&row.best_share);
            ui.end_row();
        }
    });
    for note in &report.notes {
        ui.weak(note);
    }
}

pub fn strategy_rows(form: &mut Form, ui: &mut egui::Ui, pick: &mut Option<PickTarget>) {
    ui.separator();
    ui.label(
            "Strategy per seat (a name, `adaptive:reading,tempo`, `neat:PATH`, `endgame-solver:max-unseen=6`):",
        );
    for (seat, spec) in form.strategies.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(format!("Seat {}", seat + 1));
            ui.add(egui::TextEdit::singleline(spec).desired_width(320.0));
            egui::ComboBox::from_id_salt(seat)
                .selected_text("pick")
                .show_ui(ui, |ui| {
                    for name in STRATEGIES {
                        ui.selectable_value(spec, name.to_owned(), name);
                    }
                });
            if ui.button("Genome file").clicked() {
                *pick = Some(PickTarget::Genome(seat));
            }
        });
    }
}

pub fn extras_row(form: &mut Form, ui: &mut egui::Ui) {
    ui.separator();
    ui.horizontal(|ui| {
        ui.checkbox(&mut form.explain, "Explain every statistic");
        let mut passes = form.useful_passes.is_some();
        ui.checkbox(&mut passes, "Useful-passes analysis");
        if passes != form.useful_passes.is_some() {
            form.useful_passes = passes.then_some(20);
        }
        if let Some(rollouts) = &mut form.useful_passes {
            ui.add(
                egui::DragValue::new(rollouts)
                    .range(1..=1000)
                    .suffix(" rollouts"),
            );
            ui.add(
                egui::DragValue::new(&mut form.useful_pass_sample)
                    .range(0.0..=1.0)
                    .speed(0.01)
                    .prefix("sample "),
            );
            ui.add(
                egui::DragValue::new(&mut form.useful_pass_margin)
                    .range(0.0..=f64::MAX)
                    .speed(0.01)
                    .prefix("margin "),
            );
        }
    });
}

pub fn combo<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut T,
    all: &[T],
    name: fn(T) -> &'static str,
) {
    ui.label(label);
    egui::ComboBox::from_id_salt(label)
        .selected_text(name(*value))
        .show_ui(ui, |ui| {
            for &option in all {
                ui.selectable_value(value, option, name(option));
            }
        });
}

pub fn number<T: egui::emath::Numeric>(ui: &mut egui::Ui, label: &str, value: &mut T) {
    ui.label(label);
    ui.add(egui::DragValue::new(value));
}

/// Opens `url` in the default browser.
pub fn open_in_browser(url: &str) {
    let mut command = browser_command(url);
    if let Ok(mut child) = command.spawn() {
        // Reap the launcher so it does not stay a zombie until the window closes.
        std::thread::spawn(move || {
            let _ = child.wait();
        });
    }
}

#[cfg(windows)]
pub fn browser_command(url: &str) -> std::process::Command {
    use std::os::windows::process::CommandExt;
    // Not `cmd /C start`: `&` in a URL would end the command, and cmd flashes a
    // console window (the window itself has none). The URL is one argument here.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let mut c = std::process::Command::new("rundll32");
    c.args(["url.dll,FileProtocolHandler", url]);
    c.creation_flags(CREATE_NO_WINDOW);
    c
}

#[cfg(target_os = "macos")]
pub fn browser_command(url: &str) -> std::process::Command {
    let mut c = std::process::Command::new("open");
    c.arg(url);
    c
}

#[cfg(not(any(windows, target_os = "macos")))]
pub fn browser_command(url: &str) -> std::process::Command {
    let mut c = std::process::Command::new("xdg-open");
    c.arg(url);
    c
}
