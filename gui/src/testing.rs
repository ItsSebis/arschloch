//! Helpers shared by the unit tests.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A fresh folder under the system temp dir, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(label: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "arschloch-gui-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The real `cli`, built next to the test binary's directory when the whole
/// workspace is tested; `None` (with a note) when only this crate was built.
pub fn real_cli() -> Option<PathBuf> {
    let exe = std::env::current_exe().unwrap();
    let name = if cfg!(windows) { "cli.exe" } else { "cli" };
    let cli = exe
        .ancestors()
        .nth(2)
        .map(|dir| dir.join(name))
        .filter(|p| p.exists());
    if cli.is_none() {
        // CI builds the whole workspace, so a missing binary there is a real failure.
        assert!(
            std::env::var_os("CI").is_none(),
            "no `cli` binary next to the test binary"
        );
        eprintln!(
            "skipped: no `cli` binary next to the test binary (run `cargo build -p cli` first)"
        );
    }
    cli
}
