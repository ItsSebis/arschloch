//! The file picker: opening it for a text field and filling the field from the pick.

use eframe::egui;

use crate::browse::{Kind, Picker};
use crate::{App, PickTarget};
use std::path::PathBuf;

impl App {
    /// Opens the picker for `target`, starting where its text field points.
    pub(crate) fn open_picker(&mut self, target: PickTarget) {
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

    pub(crate) fn apply_pick(&mut self, target: PickTarget, path: &std::path::Path) {
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
    pub(crate) fn picker_window(&mut self, ctx: &egui::Context) {
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
}
