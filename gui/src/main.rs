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
mod tabs;
#[cfg(test)]
mod testing;
mod widgets;

use std::fs::OpenOptions;
use std::io::Write as _;
use std::time::{Duration, Instant};

use advisor::{AdvisorForm, Recommendation};
use browse::Picker;
use eframe::egui;
use form::Form;
use job::{cli_path, Job};
use progress::Progress;
use solver::{Place, Report, SolverForm};
use widgets::open_in_browser;

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
