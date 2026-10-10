//! Timing and the parallel pass with progress reports used by the stages.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;

/// `f()` and the wall seconds it took.
pub(super) fn timed<R>(f: impl FnOnce() -> R) -> (R, f64) {
    let stage = Instant::now();
    let value = f();
    (value, stage.elapsed().as_secs_f64())
}

/// The training evaluation reports progress in this many steps (the
/// observer sees `done` at multiples of `ceil(total / PROGRESS_STEPS)`,
/// then `total`).
pub(super) const PROGRESS_STEPS: usize = 10;

/// Maps `work` over `items` as one parallel pass (rayon tasks, results
/// in item order) while the calling thread, which is the only one that may
/// touch the observer, calls `report(done)` at the progress steps: the
/// multiples of `ceil(len / PROGRESS_STEPS)` and finally `len`, each once,
/// in order, as soon as that many items are finished.
///
/// The pass is spawned into an `in_place_scope`, so the caller keeps its
/// own thread; if the caller is itself a worker of the pool (for example
/// inside `ThreadPool::install`) it lends that thread to the pool between
/// polls via `yield_now`, so a one-thread pool still makes progress (its
/// progress then arrives in fewer, later steps). A caller outside the pool
/// sleeps briefly between polls.
pub(super) fn run_with_progress<T, R>(
    items: &[T],
    work: impl Fn(&T) -> R + Sync,
    mut report: impl FnMut(usize),
) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    let total = items.len();
    let batch = total.div_ceil(PROGRESS_STEPS).max(1);
    let done = AtomicUsize::new(0);
    let ended = AtomicBool::new(false);
    let mut results = Vec::new();
    rayon::in_place_scope(|scope| {
        scope.spawn(|_| {
            // Also set when `work` panics, so the polling loop cannot spin
            // forever (the scope then re-raises the panic).
            struct Ended<'a>(&'a AtomicBool);
            impl Drop for Ended<'_> {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::Release);
                }
            }
            let _ended = Ended(&ended);
            results = items
                .par_iter()
                .map(|item| {
                    let value = work(item);
                    done.fetch_add(1, Ordering::Release);
                    value
                })
                .collect();
        });
        let mut reported = 0;
        while reported < total {
            let over = ended.load(Ordering::Acquire);
            let finished = done.load(Ordering::Acquire);
            if over && finished < total {
                break;
            }
            let mut progressed = false;
            while reported < total && finished >= (reported + batch).min(total) {
                reported = (reported + batch).min(total);
                report(reported);
                progressed = true;
            }
            if !progressed && !matches!(rayon::yield_now(), Some(rayon::Yield::Executed)) {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
    });
    results
}
