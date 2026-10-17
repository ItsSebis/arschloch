//! The file picker's model: what a folder holds and where "up" goes. No
//! drawing code, so it is unit-tested; `main.rs` draws it.

use std::path::{Path, PathBuf};

/// What the picker chooses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// An existing file, optionally only with one of the extensions.
    OpenFile,
    /// A file name to write (typed, in the shown folder).
    SaveFile,
    /// The shown folder itself.
    Folder,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub is_dir: bool,
}

/// The contents of one folder, folders first, each group sorted by name
/// (case-insensitively); hidden entries (a leading dot) are left out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listing {
    pub dir: PathBuf,
    pub entries: Vec<Entry>,
    /// Why the folder cannot be read; the entries are then empty.
    pub error: Option<String>,
}

impl Listing {
    /// Lists `dir`. In `OpenFile` and `SaveFile` mode files are kept only when
    /// their extension is in `extensions` (empty: every file); `Folder` mode
    /// shows folders only.
    pub fn read(dir: &Path, kind: Kind, extensions: &[&str]) -> Self {
        let mut entries = Vec::new();
        let mut error = None;
        match std::fs::read_dir(dir) {
            Ok(iter) => {
                for item in iter.flatten() {
                    let name = item.file_name().to_string_lossy().into_owned();
                    if name.starts_with('.') {
                        continue;
                    }
                    // `metadata` follows symlinks, so a link to a folder opens like one.
                    let is_dir = item.path().metadata().is_ok_and(|m| m.is_dir());
                    if is_dir || (kind != Kind::Folder && extension_ok(&name, extensions)) {
                        entries.push(Entry { name, is_dir });
                    }
                }
            }
            Err(e) => error = Some(format!("cannot read {}: {e}", dir.display())),
        }
        entries.sort_by_cached_key(|e| (!e.is_dir, e.name.to_lowercase()));
        Self {
            dir: dir.to_path_buf(),
            entries,
            error,
        }
    }
}

fn extension_ok(name: &str, extensions: &[&str]) -> bool {
    extensions.is_empty()
        || Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| extensions.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// The folder a picker starts in for the text already typed into the field:
/// that path's folder when it exists, else the current directory.
pub fn start_dir(typed: &str) -> PathBuf {
    let path = Path::new(typed.trim());
    let candidate = if path.is_dir() {
        Some(path)
    } else {
        path.parent().filter(|p| !p.as_os_str().is_empty())
    };
    candidate
        .filter(|p| p.is_dir())
        .and_then(|p| std::fs::canonicalize(p).ok())
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The file name typed into a save picker, checked: not empty, no folder part.
///
/// # Errors
///
/// What is wrong with the name, in words.
pub fn save_name(name: &str) -> Result<&str, &'static str> {
    let name = name.trim();
    if name.is_empty() {
        Err("type a file name")
    } else if name.contains(['/', '\\']) || name == "." || name == ".." {
        Err("a file name has no folder part (use the folder list)")
    } else {
        Ok(name)
    }
}

/// An open picker: where it is, what it picks and which text field gets the result.
#[derive(Clone, Debug)]
pub struct Picker<T> {
    pub target: T,
    pub kind: Kind,
    pub extensions: Vec<&'static str>,
    pub listing: Listing,
    /// The typed file name of a save picker.
    pub name: String,
}

impl<T> Picker<T> {
    pub fn open(target: T, kind: Kind, extensions: &[&'static str], typed: &str) -> Self {
        let dir = start_dir(typed);
        let typed = Path::new(typed.trim());
        let name = if kind == Kind::SaveFile && !typed.is_dir() {
            typed
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            String::new()
        };
        Self {
            target,
            kind,
            extensions: extensions.to_vec(),
            listing: Listing::read(&dir, kind, extensions),
            name,
        }
    }

    pub fn go(&mut self, dir: &Path) {
        self.listing = Listing::read(dir, self.kind, &self.extensions);
    }

    /// Goes one folder up (stays put at the root).
    pub fn up(&mut self) {
        if let Some(parent) = self.listing.dir.parent().map(Path::to_path_buf) {
            self.go(&parent);
        }
    }

    /// Enters the sub-folder `name` of the shown folder.
    pub fn enter(&mut self, name: &str) {
        let dir = self.listing.dir.join(name);
        self.go(&dir);
    }

    /// The path the picker would return now: the shown folder (folder mode),
    /// the typed name in it (save mode); open mode returns a clicked file via
    /// `file`.
    ///
    /// # Errors
    ///
    /// A save name that is not valid.
    pub fn result(&self) -> Result<PathBuf, &'static str> {
        match self.kind {
            Kind::Folder | Kind::OpenFile => Ok(self.listing.dir.clone()),
            Kind::SaveFile => save_name(&self.name).map(|n| self.listing.dir.join(n)),
        }
    }

    /// The path of the file `name` in the shown folder.
    pub fn file(&self, name: &str) -> PathBuf {
        self.listing.dir.join(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TempDir;

    fn sample() -> TempDir {
        let dir = TempDir::new("browse");
        for sub in ["runs", "Alpha", ".hidden"] {
            std::fs::create_dir(dir.0.join(sub)).unwrap();
        }
        for file in ["b.json", "a.JSON", "notes.txt", ".secret.json"] {
            std::fs::write(dir.0.join(file), "x").unwrap();
        }
        dir
    }

    fn names(listing: &Listing) -> Vec<String> {
        listing
            .entries
            .iter()
            .map(|e| format!("{}{}", e.name, if e.is_dir { "/" } else { "" }))
            .collect()
    }

    #[test]
    fn folders_come_first_sorted_without_case_and_hidden_ones_are_left_out() {
        let dir = sample();
        let listing = Listing::read(&dir.0, Kind::OpenFile, &[]);
        assert_eq!(
            names(&listing),
            ["Alpha/", "runs/", "a.JSON", "b.json", "notes.txt"]
        );
        assert!(listing.error.is_none());
    }

    #[test]
    fn the_extension_filter_ignores_case_and_keeps_folders() {
        let dir = sample();
        let listing = Listing::read(&dir.0, Kind::OpenFile, &["json"]);
        assert_eq!(names(&listing), ["Alpha/", "runs/", "a.JSON", "b.json"]);
    }

    #[test]
    fn folder_mode_shows_folders_only() {
        let dir = sample();
        let listing = Listing::read(&dir.0, Kind::Folder, &[]);
        assert_eq!(names(&listing), ["Alpha/", "runs/"]);
    }

    #[test]
    fn an_unreadable_folder_is_an_error_message_not_a_panic() {
        let listing = Listing::read(Path::new("/definitely/not/here"), Kind::OpenFile, &[]);
        assert_eq!(listing.entries.len(), 0);
        assert!(listing.error.unwrap().contains("cannot read"));
    }

    #[test]
    fn the_picker_starts_where_the_typed_path_points() {
        let dir = sample();
        let typed_file = dir.0.join("runs").join("new.json");
        let picker = Picker::open((), Kind::SaveFile, &["json"], &typed_file.to_string_lossy());
        assert_eq!(
            picker.listing.dir,
            std::fs::canonicalize(dir.0.join("runs")).unwrap()
        );
        assert_eq!(
            picker.name, "new.json",
            "a save picker keeps the typed name"
        );
        let picker = Picker::open((), Kind::Folder, &[], &dir.0.to_string_lossy());
        assert_eq!(picker.listing.dir, std::fs::canonicalize(&dir.0).unwrap());
        assert_eq!(picker.name, "");
        // Nonsense falls back to the current directory.
        let picker = Picker::open((), Kind::Folder, &[], "/definitely/not/here/x.json");
        assert_eq!(picker.listing.dir, std::env::current_dir().unwrap());
    }

    #[test]
    fn up_and_enter_move_through_folders() {
        let dir = sample();
        let mut picker = Picker::open((), Kind::OpenFile, &[], &dir.0.to_string_lossy());
        let start = picker.listing.dir.clone();
        picker.enter("runs");
        assert_eq!(picker.listing.dir, start.join("runs"));
        picker.up();
        assert_eq!(picker.listing.dir, start);
        let mut root = Picker::open((), Kind::Folder, &[], "/");
        // "/" is the drive root on Windows (`\\?\D:\`), so compare with what it resolves to.
        let top = std::fs::canonicalize("/").unwrap();
        assert_eq!(root.listing.dir, top);
        root.up();
        assert_eq!(root.listing.dir, top, "the root has no parent");
    }

    #[test]
    fn what_a_picker_returns_depends_on_its_kind() {
        let dir = sample();
        let mut save = Picker::open((), Kind::SaveFile, &[], &dir.0.to_string_lossy());
        assert!(save.result().is_err(), "a save picker needs a name");
        save.name = "out.json".to_owned();
        assert_eq!(save.result().unwrap(), save.listing.dir.join("out.json"));
        save.name = "../escape.json".to_owned();
        assert!(save.result().is_err());
        let folder = Picker::open((), Kind::Folder, &[], &dir.0.to_string_lossy());
        assert_eq!(folder.result().unwrap(), folder.listing.dir);
        assert_eq!(folder.file("x.json"), folder.listing.dir.join("x.json"));
    }

    #[test]
    fn save_names_are_checked() {
        assert_eq!(save_name("  a.json "), Ok("a.json"));
        for bad in ["", "   ", "a/b", "a\\b", ".", ".."] {
            assert!(save_name(bad).is_err(), "{bad:?}");
        }
    }
}
