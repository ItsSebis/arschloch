//! The simulation, command and browser tabs.

use eframe::egui;

use crate::form::{Deck, DuplicateRule, ExchangeRule, Form, PassRule, SkillScore};
use crate::job::split_args;
use crate::widgets::{combo, extras_row, number, strategy_rows};
use crate::{App, PickTarget};

impl App {
    pub(crate) fn simulation_tab(&mut self, ui: &mut egui::Ui) {
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

    pub(crate) fn preset_row(&mut self, ui: &mut egui::Ui) {
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

    pub(crate) fn command_tab(&mut self, ui: &mut egui::Ui) {
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

    pub(crate) fn browser_tab(&mut self, ui: &mut egui::Ui) {
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
