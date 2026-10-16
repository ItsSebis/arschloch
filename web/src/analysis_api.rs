//! `GET /api/analysis`: the run analysis (`sim::training::analyse`) of the
//! runs under the dashboard's root, as JSON for the analysis page.
//!
//! The reply is `{"runs": [{label, analysis, series}], "skipped_runs":
//! [text]}` (runs of a set directory that could not be read are listed, the
//! others are analysed). Equal labels get their parent directory in front.
//!
//! `?path=` narrows it to a file or directory, but only inside the root:
//! the path is resolved against the root, `..` and anything that resolves
//! (after following symlinks) outside it is refused. Run directories and
//! event files of a set are canonicalised too and left out when they
//! resolve outside the root, and error text never holds an absolute path.

use std::path::{Component, Path, PathBuf};

use serde_json::json;
use sim::training::{analyse, load_runs_within, qualify_labels, series, AnalysisError, Loaded};

use crate::routes::Response;

/// Percent-decoding of a query value (`+` is left alone: paths may hold it).
fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = text.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Why a `?path=` was not usable.
enum PathError {
    /// Outside the root, or not a plain path.
    Refused,
    /// Inside the root but not there.
    Missing,
}

/// The canonical path a `?path=` value names, guaranteed inside `root`.
fn resolve(root: &Path, wanted: &str) -> Result<PathBuf, PathError> {
    let wanted = Path::new(wanted);
    if wanted.components().any(|c| c == Component::ParentDir) || wanted.as_os_str().is_empty() {
        return Err(PathError::Refused);
    }
    let root = root.canonicalize().map_err(|_| PathError::Missing)?;
    // An absolute path replaces the root in `join`; the containment check
    // below then decides.
    let joined = root.join(wanted);
    let canonical = joined.canonicalize().map_err(|_| {
        // Not there: refused if it would not be inside the root anyway.
        if joined.starts_with(&root) {
            PathError::Missing
        } else {
            PathError::Refused
        }
    })?;
    if canonical.starts_with(&root) {
        Ok(canonical)
    } else {
        Err(PathError::Refused)
    }
}

pub(crate) fn analysis(root: &Path, query: &str) -> Response {
    let target = match super::routes::query_value(query, "path") {
        None => root.to_path_buf(),
        Some(raw) => {
            let Some(wanted) = percent_decode(raw) else {
                return Response::error(400, "path is not valid percent-encoded text");
            };
            match resolve(root, &wanted) {
                Ok(path) => path,
                Err(PathError::Refused) => {
                    return Response::error(
                        400,
                        "path must be a file or directory inside the served run directory",
                    );
                }
                Err(PathError::Missing) => {
                    return Response::error(
                        404,
                        "no such file or directory under the served run directory",
                    );
                }
            }
        }
    };
    match load_runs_within(&target, Some(root)) {
        Ok(Loaded {
            mut runs,
            skipped_runs,
        }) => {
            qualify_labels(&mut runs, false);
            let runs: Vec<serde_json::Value> = runs
                .iter()
                .map(|run| {
                    json!({
                        "label": run.label,
                        "analysis": analyse(run),
                        "series": series(run),
                    })
                })
                .collect();
            Response::json(200, &json!({ "runs": runs, "skipped_runs": skipped_runs }))
        }
        // Nothing to analyse there (yet): like a missing path.
        Err(error @ AnalysisError::NoEvents(_)) => Response::error(404, &error.to_string()),
        // A file that exists but cannot be read is a server-side problem. The
        // loader's text names paths relative to the root only.
        Err(error) => Response::error(500, &error.to_string()),
    }
}
