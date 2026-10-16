//! The training tab (advisor and the training form).

use eframe::egui;

use crate::advisor::Preset;
use crate::form::Deck;
use crate::widgets::{combo, number, recommendation_editor};
use crate::{App, Capture, PickTarget};

impl App {
    pub(crate) fn training_tab(&mut self, ui: &mut egui::Ui) {
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
