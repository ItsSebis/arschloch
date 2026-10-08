//! An incremental reader of a run's `events.jsonl`.
//!
//! The trainer appends one JSON line per event, and on resume rewrites
//! the file (trimming events after the checkpoint). The index therefore:
//! reads only the bytes added since the last refresh; keeps a partial
//! last line unread until its newline arrives; skips lines that do not
//! parse; and starts over (bumping `epoch`) when the file shrinks or is
//! replaced, so a client can tell its cached events are stale.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use sim::training::{Event, GenerationEvent, RunStart};

pub struct EventIndex {
    path: PathBuf,
    offset: u64,
    identity: Option<u64>,
    epoch: u64,
    generations: Vec<GenerationEvent>,
    run_start: Option<RunStart>,
    finished: bool,
}

#[cfg(unix)]
#[allow(clippy::unnecessary_wraps)] // the non-unix variant has no file identity to offer
fn identity(metadata: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(metadata.ino())
}

#[cfg(not(unix))]
fn identity(_: &std::fs::Metadata) -> Option<u64> {
    None
}

impl EventIndex {
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            offset: 0,
            identity: None,
            epoch: 0,
            generations: Vec::new(),
            run_start: None,
            finished: false,
        }
    }

    /// Bumped whenever previously returned events may no longer be valid.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    #[must_use]
    pub fn finished(&self) -> bool {
        self.finished
    }

    #[must_use]
    pub fn run_start(&self) -> Option<&RunStart> {
        self.run_start.as_ref()
    }

    #[must_use]
    pub fn generations(&self) -> &[GenerationEvent] {
        &self.generations
    }

    fn reset(&mut self) {
        let had_content = self.offset > 0 || !self.generations.is_empty();
        self.offset = 0;
        self.generations.clear();
        self.run_start = None;
        self.finished = false;
        if had_content {
            self.epoch += 1;
        }
    }

    /// Reads whatever was appended since the last call. A missing or
    /// unreadable file simply means "nothing (yet)".
    pub fn refresh(&mut self) {
        let Ok(mut file) = File::open(&self.path) else {
            self.reset();
            return;
        };
        let Ok(metadata) = file.metadata() else {
            return;
        };
        let current = identity(&metadata);
        if (self.identity.is_some() && current != self.identity) || metadata.len() < self.offset {
            self.reset();
        }
        self.identity = current;
        if metadata.len() == self.offset || file.seek(SeekFrom::Start(self.offset)).is_err() {
            return;
        }
        let mut added = Vec::new();
        if file.read_to_end(&mut added).is_err() {
            return;
        }
        // Only complete lines: the trainer may be mid-write.
        let Some(end) = added.iter().rposition(|&b| b == b'\n') else {
            return;
        };
        self.offset += (end + 1) as u64;
        for line in added[..end].split(|&b| b == b'\n') {
            if let Ok(event) = serde_json::from_slice::<Event>(line) {
                self.apply(event);
            }
        }
    }

    fn apply(&mut self, event: Event) {
        match event {
            Event::RunStart(start) => {
                self.run_start = Some(*start);
                self.finished = false;
            }
            Event::Generation(generation) => {
                match self
                    .generations
                    .iter_mut()
                    .find(|g| g.generation == generation.generation)
                {
                    Some(existing) => *existing = *generation,
                    None => self.generations.push(*generation),
                }
            }
            Event::RunEnd(_) => self.finished = true,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write;

    use super::*;
    use crate::test_fixture::{sample_events, temp_path};

    fn append(path: &std::path::Path, text: &str) {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        file.write_all(text.as_bytes()).unwrap();
    }

    #[test]
    fn a_missing_file_is_just_empty() {
        let mut index = EventIndex::new(temp_path("missing"));
        index.refresh();
        assert!(index.generations().is_empty() && index.run_start().is_none());
        assert!(!index.finished());
        assert_eq!(index.epoch(), 0);
    }

    #[test]
    fn it_reads_incrementally_and_tracks_start_and_end() {
        let lines = sample_events();
        let path = temp_path("incremental");
        let mut index = EventIndex::new(path.clone());
        append(&path, &format!("{}\n{}\n", lines[0], lines[1]));
        index.refresh();
        assert!(index.run_start().is_some());
        assert_eq!(index.generations().len(), 1);
        assert!(!index.finished());
        append(
            &path,
            &format!("{}\n{}\n", lines[2], lines[lines.len() - 1]),
        );
        index.refresh();
        assert_eq!(index.generations().len(), 2);
        assert!(index.finished(), "the last line is the run end");
        assert_eq!(index.epoch(), 0, "appends never bump the epoch");
        fs::remove_file(path).ok();
    }

    #[test]
    fn a_partial_last_line_waits_for_its_newline() {
        let lines = sample_events();
        let path = temp_path("partial");
        let mut index = EventIndex::new(path.clone());
        let half = lines[1].len() / 2;
        append(&path, &format!("{}\n{}", lines[0], &lines[1][..half]));
        index.refresh();
        assert!(
            index.generations().is_empty(),
            "half a line is not an event"
        );
        append(&path, &format!("{}\n", &lines[1][half..]));
        index.refresh();
        assert_eq!(index.generations().len(), 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn garbage_lines_are_skipped_not_fatal() {
        let lines = sample_events();
        let path = temp_path("garbage");
        append(
            &path,
            &format!(
                "{}\nnot json at all\n{{\"type\":\"nonsense\"}}\n{}\n",
                lines[0], lines[1]
            ),
        );
        let mut index = EventIndex::new(path.clone());
        index.refresh();
        assert_eq!(index.generations().len(), 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn a_rewritten_file_starts_over_and_bumps_the_epoch() {
        let lines = sample_events();
        let path = temp_path("rewrite");
        append(
            &path,
            &format!("{}\n{}\n{}\n", lines[0], lines[1], lines[2]),
        );
        let mut index = EventIndex::new(path.clone());
        index.refresh();
        assert_eq!(index.generations().len(), 2);
        // The trainer's resume rewrites the file (shorter, new inode).
        let replacement = path.with_extension("tmp");
        fs::write(&replacement, format!("{}\n{}\n", lines[0], lines[1])).unwrap();
        fs::rename(&replacement, &path).unwrap();
        index.refresh();
        assert_eq!(index.epoch(), 1);
        assert_eq!(index.generations().len(), 1);
        fs::remove_file(path).ok();
    }

    #[test]
    fn a_replacement_that_grew_past_the_old_offset_is_still_detected() {
        let lines = sample_events();
        let path = temp_path("regrew");
        append(&path, &format!("{}\n{}\n", lines[0], lines[1]));
        let mut index = EventIndex::new(path.clone());
        index.refresh();
        let replacement = path.with_extension("tmp");
        fs::write(
            &replacement,
            format!("{}\n{}\n{}\n{}\n", lines[0], lines[1], lines[2], lines[3]),
        )
        .unwrap();
        fs::rename(&replacement, &path).unwrap();
        index.refresh();
        assert_eq!(index.epoch(), 1, "same-or-longer replacement is a new file");
        assert_eq!(index.generations().len(), 3);
        fs::remove_file(path).ok();
    }

    #[test]
    fn a_repeated_generation_replaces_the_earlier_one() {
        let lines = sample_events();
        let path = temp_path("repeat");
        append(
            &path,
            &format!("{}\n{}\n{}\n", lines[0], lines[1], lines[1]),
        );
        let mut index = EventIndex::new(path.clone());
        index.refresh();
        assert_eq!(index.generations().len(), 1);
        fs::remove_file(path).ok();
    }
}
