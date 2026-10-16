//! A small desktop window to configure and run simulations and to read what
//! the program is doing. It runs the real `cli` program with the arguments the
//! form builds, so it behaves exactly like the command line; the chart-heavy
//! parts (play against the models, dashboards) stay in the browser and are
//! started from here. See docs/GUI.md.
// A release build on Windows is a window program: no console behind it.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod advisor;
mod browse;
mod form;
mod job;
mod progress;
mod solver;
#[cfg(test)]
mod testing;
mod widgets;

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use advisor::{AdvisorForm, Preset, Recommendation};
use browse::{Kind, Picker};
use eframe::egui;
use form::{Deck, DuplicateRule, ExchangeRule, Form, PassRule, SkillScore};
use job::{cli_path, split_args, Job};
use progress::Progress;
use solver::{Place, Report, SolverForm};
use widgets::{
    card_grid, combo, extras_row, number, open_in_browser, recommendation_editor, seat_rows,
    solver_report, strategy_rows,
};

/// Lines kept in the window (the log file keeps everything).
const LOG_LINES: usize = 5000;
const LOG_FILE: &str = "arschloch-gui.log";
const STRATEGIES: [&str; 8] = [
    "lowest-legal",
    "greedy-highest",
    "random-legal",
    "hold-back-pairs",
    "card-counter",
    "endgame-denial",
    "adaptive",
    "endgame-solver",
];

#[derive(Debug, PartialEq)]
enum Tab {
    Simulation,
    Solver,
    Training,
    Command,
    Browser,
}

impl Tab {
    /// The tab a `--tab NAME` argument asks for.
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "simulation" => Some(Self::Simulation),
            "solver" => Some(Self::Solver),
            "training" => Some(Self::Training),
            "command" => Some(Self::Command),
            "browser" => Some(Self::Browser),
            _ => None,
        }
    }
}

/// The text field a file picker fills in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PickTarget {
    Output,
    Preset,
    /// The strategy spec of a seat: becomes `neat:PATH`.
    Genome(usize),
    WatchDir,
    AdvisorOut,
    TrainOut,
}

/// What a running job's output is collected for, besides the log.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Capture {
    Solve,
    Advice,
}

const POSITION_FILE: &str = "arschloch-gui-position.txt";

struct App {
    tab: Tab,
    form: Form,
    preset_path: String,
    command: String,
    watch_dir: String,
    log: Vec<String>,
    started: Instant,
    job: Option<Job>,
    /// Open the first `http://` address the running job prints.
    open_url: bool,
    picker: Option<Picker<PickTarget>>,
    /// A Browse button was pressed; `ui` opens the picker after drawing.
    pick_request: Option<PickTarget>,
    /// The last progress line of the running job.
    progress: Option<Progress>,
    job_started: Instant,
    capture: Option<Capture>,
    captured: Vec<String>,
    solver: SolverForm,
    brush: Place,
    solved: Option<Result<Report, String>>,
    advisor: AdvisorForm,
    advice: Option<String>,
    recommendation: Option<Result<Recommendation, String>>,
}

impl App {
    fn new() -> Self {
        Self {
            tab: Tab::Simulation,
            form: Form::default(),
            preset_path: "arschloch-gui-preset.json".to_owned(),
            command: String::new(),
            watch_dir: String::new(),
            log: vec!["Ready. Choose a tab, set the options and press Run.".to_owned()],
            started: Instant::now(),
            job: None,
            open_url: false,
            picker: None,
            pick_request: None,
            progress: None,
            job_started: Instant::now(),
            capture: None,
            captured: Vec::new(),
            solver: SolverForm::example(),
            brush: Place::Unseen,
            solved: None,
            advisor: AdvisorForm::default(),
            advice: None,
            recommendation: None,
        }
    }

    /// Adds a line to the log and to the log file.
    fn say(&mut self, text: impl Into<String>) {
        let secs = self.started.elapsed().as_secs();
        let line = format!("[{:02}:{:02}] {}", secs / 60, secs % 60, text.into());
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(LOG_FILE) {
            let _ = writeln!(file, "{line}");
        }
        self.log.push(line);
        if self.log.len() > LOG_LINES {
            self.log.drain(..self.log.len() - LOG_LINES);
        }
    }

    fn start(&mut self, args: &[String], open_url: bool) {
        self.start_capturing(args, open_url, None);
    }

    /// Starts a job; with `capture` its output is also collected for the tab that asked.
    fn start_capturing(&mut self, args: &[String], open_url: bool, capture: Option<Capture>) {
        self.say(format!("$ cli {}", args.join(" ")));
        match Job::spawn(&cli_path(), args) {
            Ok(job) => {
                self.job = Some(job);
                self.open_url = open_url;
                self.progress = None;
                self.job_started = Instant::now();
                self.capture = capture;
                self.captured.clear();
            }
            Err(error) => self.say(format!(
                "cannot start {}: {error} (the gui looks for `cli` next to itself, then on the PATH)",
                cli_path().display()
            )),
        }
    }

    fn poll(&mut self) {
        let lines = self.job.as_mut().map(Job::poll).unwrap_or_default();
        for line in lines {
            if let Some(progress) = progress::parse(&line) {
                self.progress = Some(progress);
                continue; // the bar shows it; hundreds of lines would drown the log
            }
            if self.capture.is_some() {
                self.captured.push(line.clone());
            }
            if line.starts_with("[finished") || line.starts_with("[cancelled") {
                self.progress = None;
                self.finish_capture();
            }
            if self.open_url {
                if let Some(at) = line.find("http://") {
                    self.open_url = false;
                    open_in_browser(line[at..].split_whitespace().next().unwrap_or(""));
                }
            }
            self.say(line);
        }
    }

    /// The job ended: hand its collected output to the tab that asked for it.
    fn finish_capture(&mut self) {
        let text = self.captured.join("\n");
        match self.capture.take() {
            Some(Capture::Solve) => self.solved = Some(solver::parse_report(&text)),
            Some(Capture::Advice) => {
                self.recommendation = Some(
                    advisor::find_command(&text)
                        .ok_or_else(|| {
                            "the advisor did not print a recommendation (see the log)".to_owned()
                        })
                        .and_then(Recommendation::parse),
                );
                self.advice = Some(text);
            }
            None => {}
        }
        self.captured.clear();
    }

    fn running(&self) -> bool {
        self.job.as_ref().is_some_and(Job::is_running)
    }

    fn simulation_tab(&mut self, ui: &mut egui::Ui) {
        let mut pick = None;
        let form = &mut self.form;
        egui::Grid::new("table").num_columns(4).show(ui, |ui| {
            ui.label("Players");
            let mut players = form.player_count;
            ui.add(egui::DragValue::new(&mut players).range(3..=6));
            if players != form.player_count {
                form.set_player_count(players);
            }
            combo(ui, "Deck", &mut form.deck_variant, Deck::ALL, Deck::as_str);
            ui.end_row();
            combo(
                ui,
                "Duplicate cards",
                &mut form.duplicate_rule,
                DuplicateRule::ALL,
                DuplicateRule::as_str,
            );
            combo(
                ui,
                "Pass rule",
                &mut form.pass_rule,
                PassRule::ALL,
                PassRule::as_str,
            );
            ui.end_row();
            combo(
                ui,
                "Exchange",
                &mut form.exchange_rule,
                ExchangeRule::ALL,
                ExchangeRule::as_str,
            );
            combo(
                ui,
                "Skill score",
                &mut form.skill_score,
                SkillScore::ALL,
                SkillScore::as_str,
            );
            ui.end_row();
            number(ui, "Matches", &mut form.matches);
            number(ui, "Rounds", &mut form.rounds);
            ui.end_row();
            number(ui, "Seed", &mut form.seed);
            number(ui, "Threads (0 = auto)", &mut form.threads);
            ui.end_row();
            number(ui, "Bootstrap resamples", &mut form.bootstrap_resamples);
            ui.label("Output file");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut form.output);
                if ui.button("Browse").clicked() {
                    pick = Some(PickTarget::Output);
                }
            });
            ui.end_row();
        });
        strategy_rows(form, ui, &mut pick);
        extras_row(form, ui);
        self.pick_request = pick.or(self.pick_request);
        self.preset_row(ui);
        let problem = self.form.problem();
        if let Some(problem) = &problem {
            ui.colored_label(egui::Color32::LIGHT_RED, problem);
        }
        ui.horizontal(|ui| {
            let runnable = problem.is_none() && !self.running();
            if ui
                .add_enabled(runnable, egui::Button::new("Run simulation"))
                .clicked()
            {
                self.start(&self.form.args(), false);
            }
            ui.monospace(format!("cli {}", self.form.args().join(" ")));
        });
    }

    fn preset_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Preset");
            ui.text_edit_singleline(&mut self.preset_path);
            if ui.button("Browse").clicked() {
                self.pick_request = Some(PickTarget::Preset);
            }
            if ui.button("Save").clicked() {
                let result = serde_json::to_string_pretty(&self.form)
                    .map_err(|e| e.to_string())
                    .and_then(|json| {
                        std::fs::write(&self.preset_path, json).map_err(|e| e.to_string())
                    });
                self.say(match result {
                    Ok(()) => format!("preset saved to {}", self.preset_path),
                    Err(error) => format!("cannot save the preset: {error}"),
                });
            }
            if ui.button("Load").clicked() {
                let result = std::fs::read_to_string(&self.preset_path)
                    .map_err(|e| e.to_string())
                    .and_then(|text| {
                        serde_json::from_str::<Form>(&text).map_err(|e| e.to_string())
                    });
                match result {
                    Ok(form) => {
                        self.form = form;
                        self.form.set_player_count(self.form.player_count);
                        self.say(format!("preset loaded from {}", self.preset_path));
                    }
                    Err(error) => self.say(format!("cannot load the preset: {error}")),
                }
            }
        });
    }

    fn command_tab(&mut self, ui: &mut egui::Ui) {
        ui.label(
            "Any `cli` command, for example `train --help`, `evaluate ...`, `solve docs/examples/solve-endgame.txt`:",
        );
        ui.text_edit_singleline(&mut self.command);
        if ui
            .add_enabled(!self.running(), egui::Button::new("Run"))
            .clicked()
        {
            self.start(&split_args(&self.command), false);
        }
    }

    fn browser_tab(&mut self, ui: &mut egui::Ui) {
        ui.label("Chart-heavy parts open in your browser; the server runs until you press Cancel.");
        let idle = !self.running();
        if ui
            .add_enabled(idle, egui::Button::new("Play against the models"))
            .clicked()
        {
            self.start(&["play".to_owned()], true);
        }
        if ui
            .add_enabled(
                idle,
                egui::Button::new("View the results of the last simulation"),
            )
            .clicked()
        {
            self.start(&["view".to_owned(), self.form.output.clone()], true);
        }
        ui.horizontal(|ui| {
            ui.label("Run folder");
            ui.text_edit_singleline(&mut self.watch_dir);
            if ui.button("Browse").clicked() {
                self.pick_request = Some(PickTarget::WatchDir);
            }
            let ready = idle && !self.watch_dir.trim().is_empty();
            if ui
                .add_enabled(ready, egui::Button::new("Dashboard and run analysis"))
                .clicked()
            {
                self.start(
                    &["watch".to_owned(), self.watch_dir.trim().to_owned()],
                    true,
                );
            }
        });
    }
}

impl App {
    /// Opens the picker for `target`, starting where its text field points.
    fn open_picker(&mut self, target: PickTarget) {
        let (kind, extensions, typed): (Kind, &[&'static str], String) = match target {
            PickTarget::Output => (Kind::SaveFile, &["json"], self.form.output.clone()),
            PickTarget::Preset => (Kind::OpenFile, &["json"], self.preset_path.clone()),
            PickTarget::Genome(seat) => {
                let spec = self.form.strategies.get(seat).cloned().unwrap_or_default();
                let path = spec.strip_prefix("neat:").unwrap_or("").to_owned();
                (Kind::OpenFile, &["json"], path)
            }
            PickTarget::WatchDir => (Kind::Folder, &[], self.watch_dir.clone()),
            PickTarget::AdvisorOut => (Kind::Folder, &[], self.advisor.out.clone()),
            PickTarget::TrainOut => {
                let out = match &self.recommendation {
                    Some(Ok(rec)) => rec.out.clone(),
                    _ => String::new(),
                };
                (Kind::Folder, &[], out)
            }
        };
        self.picker = Some(Picker::open(target, kind, extensions, &typed));
    }

    fn apply_pick(&mut self, target: PickTarget, path: &std::path::Path) {
        let text = path.to_string_lossy().into_owned();
        match target {
            PickTarget::Output => self.form.output = text,
            PickTarget::Preset => self.preset_path = text,
            PickTarget::Genome(seat) => {
                if let Some(spec) = self.form.strategies.get_mut(seat) {
                    *spec = format!("neat:{text}");
                }
            }
            PickTarget::WatchDir => self.watch_dir = text,
            PickTarget::AdvisorOut => self.advisor.out = text,
            PickTarget::TrainOut => {
                if let Some(Ok(rec)) = &mut self.recommendation {
                    rec.out = text;
                }
            }
        }
    }

    /// The file picker window; returns when the user chose or cancelled.
    fn picker_window(&mut self, ctx: &egui::Context) {
        let Some(mut picker) = self.picker.take() else {
            return;
        };
        let mut chosen: Option<PathBuf> = None;
        let mut open = true;
        let mut cancel = false;
        let title = match picker.kind {
            Kind::OpenFile => "Choose a file",
            Kind::SaveFile => "Choose where to save",
            Kind::Folder => "Choose a folder",
        };
        egui::Window::new(title)
            .collapsible(false)
            .default_width(520.0)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Up").clicked() {
                        picker.up();
                    }
                    ui.monospace(picker.listing.dir.display().to_string());
                });
                if let Some(error) = &picker.listing.error {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }
                let mut enter: Option<String> = None;
                egui::ScrollArea::vertical()
                    .max_height(280.0)
                    .auto_shrink(false)
                    .show(ui, |ui| {
                        for entry in &picker.listing.entries {
                            if entry.is_dir {
                                if ui.button(format!("[ {} ]", entry.name)).clicked() {
                                    enter = Some(entry.name.clone());
                                }
                            } else if ui.selectable_label(false, &entry.name).clicked() {
                                if picker.kind == Kind::SaveFile {
                                    picker.name.clone_from(&entry.name);
                                } else {
                                    chosen = Some(picker.file(&entry.name));
                                }
                            }
                        }
                    });
                if let Some(name) = enter {
                    picker.enter(&name);
                }
                ui.separator();
                ui.horizontal(|ui| {
                    match picker.kind {
                        Kind::OpenFile => {
                            ui.label("Click a file to choose it.");
                        }
                        Kind::SaveFile => {
                            ui.label("File name");
                            ui.text_edit_singleline(&mut picker.name);
                            match picker.result() {
                                Ok(path) => {
                                    if ui.button("Use this name").clicked() {
                                        chosen = Some(path);
                                    }
                                }
                                Err(why) => {
                                    ui.colored_label(egui::Color32::LIGHT_RED, why);
                                }
                            }
                        }
                        Kind::Folder => {
                            if ui.button("Use this folder").clicked() {
                                chosen = picker.result().ok();
                            }
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });
        if let Some(path) = chosen {
            self.apply_pick(picker.target, &path);
        } else if open && !cancel {
            self.picker = Some(picker);
        }
    }

    fn solver_tab(&mut self, ui: &mut egui::Ui) {
        ui.label(
            "Describe a real situation, press Solve, and the real `cli solve` ranks the moves by the exact expected \
             finishing place. Pick a brush, then click cards: the whole deck starts as played.",
        );
        let form = &mut self.solver;
        ui.horizontal(|ui| {
            let mut players = form.players;
            ui.label("Players");
            ui.add(egui::DragValue::new(&mut players).range(3..=6));
            if players != form.players {
                form.set_players(players);
            }
            ui.label("My seat");
            ui.add(egui::DragValue::new(&mut form.seat).range(0..=form.players - 1));
            if ui.button("Load the example").clicked() {
                *form = SolverForm::example();
                self.solved = None;
            }
        });
        let form = &mut self.solver;
        ui.horizontal(|ui| {
            ui.label("Brush");
            for place in Place::ALL {
                ui.selectable_value(&mut self.brush, place, place.label());
            }
            if ui.button("Everything played").clicked() {
                form.fill(Place::Played);
            }
            if ui.button("Everything unseen").clicked() {
                form.fill(Place::Unseen);
            }
        });
        card_grid(ui, form, self.brush);
        ui.label(format!(
            "my hand {}, on the table {}, unseen {}, played {}",
            form.count(Place::Hand),
            form.count(Place::Table),
            form.count(Place::Unseen),
            form.count(Place::Played)
        ));
        ui.separator();
        seat_rows(ui, form);
        let problem = form.problem();
        if let Some(problem) = &problem {
            ui.colored_label(egui::Color32::LIGHT_RED, problem);
        }
        let runnable = problem.is_none() && !self.running();
        if ui
            .add_enabled(runnable, egui::Button::new("Solve"))
            .clicked()
        {
            let path = std::env::temp_dir().join(POSITION_FILE);
            match std::fs::write(&path, self.solver.text()) {
                Ok(()) => {
                    self.solved = None;
                    let args = self.solver.args(&path.to_string_lossy());
                    self.start_capturing(&args, false, Some(Capture::Solve));
                }
                Err(error) => self.say(format!("cannot write the position file: {error}")),
            }
        }
        match &self.solved {
            Some(Ok(report)) => solver_report(ui, report),
            Some(Err(error)) => {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }
            None => {}
        }
        egui::CollapsingHeader::new("Position file (the same text `cli solve FILE` reads)").show(
            ui,
            |ui| {
                ui.monospace(self.solver.text());
            },
        );
    }

    fn training_tab(&mut self, ui: &mut egui::Ui) {
        ui.label(
            "The advisor probes this machine and the opponents (a minute or so), then recommends the settings of \
             `cli train`. Edit them, then start the training.",
        );
        let form = &mut self.advisor;
        let mut pick = None;
        egui::Grid::new("advisor").num_columns(4).show(ui, |ui| {
            ui.label("Players");
            let mut players = form.players;
            ui.add(egui::DragValue::new(&mut players).range(3..=6));
            form.players = players;
            combo(ui, "Deck", &mut form.deck, Deck::ALL, Deck::as_str);
            ui.end_row();
            combo(
                ui,
                "Budget preset",
                &mut form.preset,
                Preset::ALL,
                Preset::as_str,
            );
            ui.label("Time budget (30m, 2h)");
            ui.text_edit_singleline(&mut form.time_budget);
            ui.end_row();
            number(ui, "Probe matches", &mut form.probe_matches);
            number(ui, "Seed", &mut form.seed);
            ui.end_row();
            ui.label("Run folder");
            ui.horizontal(|ui| {
                ui.text_edit_singleline(&mut form.out);
                if ui.button("Browse").clicked() {
                    pick = Some(PickTarget::AdvisorOut);
                }
            });
            ui.label("Candidate opponents");
            ui.text_edit_singleline(&mut form.opponents).on_hover_text(
                "Strategy specs separated by spaces; empty uses the advisor's battery",
            );
            ui.end_row();
        });
        self.pick_request = pick.or(self.pick_request);
        let problem = self.advisor.problem();
        if let Some(problem) = &problem {
            ui.colored_label(egui::Color32::LIGHT_RED, problem);
        }
        let runnable = problem.is_none() && !self.running();
        if ui
            .add_enabled(runnable, egui::Button::new("Ask the advisor"))
            .clicked()
        {
            self.recommendation = None;
            self.advice = None;
            let args = self.advisor.args();
            self.start_capturing(&args, false, Some(Capture::Advice));
        }
        if let Some(advice) = &self.advice {
            egui::CollapsingHeader::new("What the advisor measured and why").show(ui, |ui| {
                ui.monospace(advice);
            });
        }
        let mut start: Option<Vec<String>> = None;
        let mut pick = None;
        let idle = !self.running();
        match &mut self.recommendation {
            Some(Ok(rec)) => recommendation_editor(ui, rec, idle, &mut pick, &mut start),
            Some(Err(error)) => {
                ui.colored_label(egui::Color32::LIGHT_RED, error.as_str());
            }
            None => {}
        }
        self.pick_request = pick.or(self.pick_request);
        if let Some(args) = start {
            self.start(&args, false);
        }
    }
}

impl eframe::App for App {
    /// Stops a running job (a server, a training) when the window closes.
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.job = None;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll();
        if self.running() {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        egui::Panel::bottom("activity")
            .resizable(true)
            .min_size(160.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Activity");
                    if ui
                        .add_enabled(self.running(), egui::Button::new("Cancel"))
                        .clicked()
                    {
                        if let Some(job) = &mut self.job {
                            job.cancel();
                        }
                    }
                    if ui.button("Clear").clicked() {
                        self.log.clear();
                    }
                    ui.label(format!("(also written to {LOG_FILE})"));
                });
                if self.running() {
                    let secs = self.job_started.elapsed().as_secs();
                    let clock = format!("{:02}:{:02}", secs / 60, secs % 60);
                    let bar = match &self.progress {
                        Some(p) => egui::ProgressBar::new(p.fraction).text(format!(
                            "{:.0}%  {}  ({clock})",
                            p.fraction * 100.0,
                            p.text
                        )),
                        // No progress lines (a plain simulation has none): an animated bar.
                        None => egui::ProgressBar::new(0.0)
                            .animate(true)
                            .text(format!("running ({clock})")),
                    };
                    ui.add(bar);
                }
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink(false)
                    .show(ui, |ui| {
                        for line in &self.log {
                            ui.monospace(line);
                        }
                    });
            });
        egui::CentralPanel::default().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Simulation, "Simulation");
                ui.selectable_value(&mut self.tab, Tab::Solver, "Solver");
                ui.selectable_value(&mut self.tab, Tab::Training, "Training");
                ui.selectable_value(&mut self.tab, Tab::Command, "Any command");
                ui.selectable_value(&mut self.tab, Tab::Browser, "In the browser");
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
                Tab::Simulation => self.simulation_tab(ui),
                Tab::Solver => self.solver_tab(ui),
                Tab::Training => self.training_tab(ui),
                Tab::Command => self.command_tab(ui),
                Tab::Browser => self.browser_tab(ui),
            });
        });
        if let Some(target) = self.pick_request.take() {
            self.open_picker(target);
        }
        let ctx = ui.ctx().clone();
        self.picker_window(&ctx);
    }
}

fn main() -> eframe::Result {
    // `arschloch-gui --tab solver` opens on that tab (simulation, solver, training, command, browser).
    let mut args = std::env::args().skip(1);
    let mut tab = Tab::Simulation;
    while let Some(arg) = args.next() {
        if arg == "--tab" {
            if let Some(found) = args.next().as_deref().and_then(Tab::from_name) {
                tab = found;
            } else {
                eprintln!(
                    "usage: arschloch-gui [--tab simulation|solver|training|command|browser]"
                );
                std::process::exit(2);
            }
        }
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 760.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Arschloch",
        options,
        Box::new(move |_cc| {
            let mut app = App::new();
            app.tab = tab;
            Ok(Box::new(app))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tab_is_found_by_name() {
        assert_eq!(Tab::from_name("solver"), Some(Tab::Solver));
        assert_eq!(Tab::from_name("browser"), Some(Tab::Browser));
        assert!(Tab::from_name("nonsense").is_none());
    }
}
