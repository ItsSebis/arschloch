//! The solver tab.

use eframe::egui;

use crate::solver::{Place, SolverForm};
use crate::widgets::{card_grid, seat_rows, solver_report};
use crate::{App, Capture, POSITION_FILE};

impl App {
    pub(crate) fn solver_tab(&mut self, ui: &mut egui::Ui) {
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
}
