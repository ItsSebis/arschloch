//! A small desktop window to configure and run simulations and to read what
//! the program is doing. It runs the real `cli` program with the arguments the
//! form builds, so it behaves exactly like the command line; the chart-heavy
//! parts (play against the models, dashboards) stay in the browser and are
//! started from here. See docs/GUI.md.

mod form;
mod job;

use std::fs::OpenOptions;
use std::io::Write as _;
use std::time::{Duration, Instant};

use eframe::egui;
use form::{Deck, DuplicateRule, ExchangeRule, Form, PassRule, SkillScore};
use job::{cli_path, split_args, Job};

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

#[derive(PartialEq)]
enum Tab {
    Simulation,
    Command,
    Browser,
}

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
        self.say(format!("$ cli {}", args.join(" ")));
        match Job::spawn(&cli_path(), args) {
            Ok(job) => {
                self.job = Some(job);
                self.open_url = open_url;
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
            if self.open_url {
                if let Some(at) = line.find("http://") {
                    self.open_url = false;
                    open_in_browser(line[at..].split_whitespace().next().unwrap_or(""));
                }
            }
            self.say(line);
        }
    }

    fn running(&self) -> bool {
        self.job.as_ref().is_some_and(Job::is_running)
    }

    fn simulation_tab(&mut self, ui: &mut egui::Ui) {
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
            ui.text_edit_singleline(&mut form.output);
            ui.end_row();
        });
        strategy_rows(form, ui);
        extras_row(form, ui);
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

fn strategy_rows(form: &mut Form, ui: &mut egui::Ui) {
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
        });
    }
}

fn extras_row(form: &mut Form, ui: &mut egui::Ui) {
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
                    .speed(0.01)
                    .prefix("margin "),
            );
        }
    });
}

fn combo<T: Copy + PartialEq>(
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

fn number<T: egui::emath::Numeric>(ui: &mut egui::Ui, label: &str, value: &mut T) {
    ui.label(label);
    ui.add(egui::DragValue::new(value));
}

/// Opens `url` in the default browser.
fn open_in_browser(url: &str) {
    let mut command = if cfg!(target_os = "windows") {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", ""]);
        c
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else {
        std::process::Command::new("xdg-open")
    };
    let _ = command.arg(url).spawn();
}

impl eframe::App for App {
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
                ui.selectable_value(&mut self.tab, Tab::Command, "Any command");
                ui.selectable_value(&mut self.tab, Tab::Browser, "In the browser");
            });
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
                Tab::Simulation => self.simulation_tab(ui),
                Tab::Command => self.command_tab(ui),
                Tab::Browser => self.browser_tab(ui),
            });
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 760.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Arschloch",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}
