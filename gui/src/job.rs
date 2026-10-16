//! Runs one command in the background and streams its output.

use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

/// A child process whose stdout and stderr arrive line by line through `poll`.
pub struct Job {
    child: Child,
    lines: Receiver<String>,
    cancelled: bool,
    done: bool,
}

/// Sends every line of `stream` to `out`; ends when the stream closes.
fn pump(stream: impl Read + Send + 'static, out: Sender<String>) {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        let mut bytes = Vec::new();
        while matches!(reader.read_until(b'\n', &mut bytes), Ok(n) if n > 0) {
            let line = String::from_utf8_lossy(&bytes).trim_end().to_owned();
            bytes.clear();
            if out.send(line).is_err() {
                break;
            }
        }
    });
}

impl Job {
    /// # Errors
    ///
    /// The program cannot be started.
    pub fn spawn(program: &Path, args: &[String]) -> io::Result<Self> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let (sender, lines) = channel();
        if let Some(stdout) = child.stdout.take() {
            pump(stdout, sender.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            pump(stderr, sender);
        }
        Ok(Self {
            child,
            lines,
            cancelled: false,
            done: false,
        })
    }

    #[cfg(test)]
    fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn is_running(&self) -> bool {
        !self.done
    }

    /// New output lines since the last call; when the process has ended, one
    /// final line says how.
    pub fn poll(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        let closed = loop {
            match self.lines.try_recv() {
                Ok(line) => out.push(line),
                Err(TryRecvError::Empty) => break false,
                Err(TryRecvError::Disconnected) => break true,
            }
        };
        if closed && !self.done {
            match self.child.try_wait() {
                Ok(None) => {}
                Ok(Some(status)) => {
                    self.done = true;
                    out.push(if self.cancelled {
                        "[cancelled]".to_owned()
                    } else {
                        status.code().map_or_else(
                            || "[finished: ended without an exit code]".to_owned(),
                            |code| format!("[finished: exit code {code}]"),
                        )
                    });
                }
                Err(error) => {
                    self.done = true;
                    out.push(format!("[finished: cannot read the exit status: {error}]"));
                }
            }
        }
        out
    }

    pub fn cancel(&mut self) {
        // A job that has already ended keeps its own exit status.
        if matches!(self.child.try_wait(), Ok(None)) {
            self.cancelled = true;
            let _ = self.child.kill();
        }
    }
}

/// Closing the window must not leave a server or a training run behind.
impl Drop for Job {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Splits a typed command line into arguments; double quotes group words.
pub fn split_args(text: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let (mut quoted, mut started) = (false, false);
    for ch in text.chars() {
        match ch {
            '"' => {
                quoted = !quoted;
                started = true;
            }
            c if c.is_whitespace() && !quoted => {
                if started {
                    args.push(std::mem::take(&mut current));
                    started = false;
                }
            }
            c => {
                current.push(c);
                started = true;
            }
        }
    }
    if started {
        args.push(current);
    }
    args
}

/// The `cli` program: next to this executable, else found on the `PATH`.
pub fn cli_path() -> PathBuf {
    let name = if cfg!(windows) { "cli.exe" } else { "cli" };
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(name)))
        .filter(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Collects the output of `job` until it has finished (or 20 s passed).
    fn drain(job: &mut Job) -> Vec<String> {
        let mut lines = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(20);
        while job.is_running() && Instant::now() < deadline {
            lines.extend(job.poll());
            std::thread::sleep(Duration::from_millis(10));
        }
        lines.extend(job.poll());
        lines
    }

    #[test]
    fn output_is_streamed_and_the_exit_is_reported() {
        let exe = std::env::current_exe().unwrap();
        let mut job = Job::spawn(&exe, &["--list".to_owned()]).unwrap();
        let lines = drain(&mut job);
        assert!(
            lines.iter().any(|l| l.contains("form::tests::")),
            "{lines:?}"
        );
        assert!(lines.last().unwrap().contains("finished"), "{lines:?}");
        assert!(
            lines.last().unwrap().contains('0'),
            "exit code 0: {lines:?}"
        );
        assert!(!job.is_running());
    }

    #[test]
    fn a_typed_command_line_splits_into_arguments() {
        assert_eq!(
            split_args(r#"train --out "my runs/a b" --seed 3"#),
            ["train", "--out", "my runs/a b", "--seed", "3"]
        );
        assert_eq!(split_args("  solve   file.txt "), ["solve", "file.txt"]);
        assert_eq!(
            split_args(r#"a "" b"#),
            ["a", "", "b"],
            "an empty quoted argument stays"
        );
        assert!(split_args("   ").is_empty());
    }

    #[test]
    fn a_missing_program_is_an_error_not_a_panic() {
        assert!(Job::spawn(std::path::Path::new("definitely-not-a-program-xyz"), &[]).is_err());
    }

    #[test]
    fn a_running_job_can_be_cancelled() {
        let exe = std::env::current_exe().unwrap();
        let args: Vec<String> = ["--ignored", "--exact", "job::tests::sleeper"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let mut job = Job::spawn(&exe, &args).unwrap();
        assert!(job.is_running());
        job.cancel();
        let lines = drain(&mut job);
        assert!(!job.is_running());
        assert!(lines.last().unwrap().contains("cancelled"), "{lines:?}");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn dropping_a_running_job_stops_its_process() {
        let exe = std::env::current_exe().unwrap();
        let args: Vec<String> = ["--ignored", "--exact", "job::tests::sleeper"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let job = Job::spawn(&exe, &args).unwrap();
        let proc = format!("/proc/{}", job.id());
        assert!(std::path::Path::new(&proc).exists());
        drop(job); // closing the window must not leave a server or a training behind
        assert!(
            !std::path::Path::new(&proc).exists(),
            "the child is still running"
        );
    }

    #[test]
    fn cancelling_a_job_that_already_ended_keeps_its_exit_status() {
        let exe = std::env::current_exe().unwrap();
        let mut job = Job::spawn(&exe, &["--list".to_owned()]).unwrap();
        std::thread::sleep(Duration::from_millis(500)); // it has exited, but was not polled yet
        job.cancel();
        let lines = drain(&mut job);
        let last = lines.last().unwrap();
        assert!(
            last.contains("finished") && !last.contains("cancelled"),
            "{lines:?}"
        );
    }

    /// The child process of `a_running_job_can_be_cancelled`.
    #[test]
    #[ignore = "only run as the child of a_running_job_can_be_cancelled"]
    fn sleeper() {
        std::thread::sleep(Duration::from_secs(60));
    }
}
