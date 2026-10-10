//! Minimal dense linear algebra for the luck estimator: ordinary least
//! squares through the normal equations, solved by Gaussian elimination
//! with partial pivoting. Sizes are tiny (a dozen columns), so clarity
//! beats numerical sophistication.

/// Why `ols` could not produce coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OlsError {
    /// No rows, no columns, or `rows`/`targets` lengths disagree.
    BadShape,
    /// Rows have differing lengths.
    RaggedRows,
}

impl std::fmt::Display for OlsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadShape => write!(f, "regression needs matching, non-empty rows and targets"),
            Self::RaggedRows => write!(f, "regression rows must all have the same length"),
        }
    }
}

impl std::error::Error for OlsError {}

/// Solves `a * x = b` (`a` square, row-major) by Gaussian elimination
/// with partial pivoting. Returns `None` when a pivot is (numerically)
/// zero, i.e. the system is singular.
#[must_use]
pub fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    let scale = a
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot_row][col].abs() < 1e-12 * scale {
            return None;
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            let pivot_tail: Vec<f64> = a[col][col..].to_vec();
            for (cell, p) in a[row][col..].iter_mut().zip(pivot_tail) {
                *cell -= factor * p;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    Some(x)
}

/// Least-squares coefficients `beta` minimising `|y - X beta|^2`, where
/// `rows` are the rows of `X` (include a constant column for an
/// intercept). If the normal equations are singular (collinear or
/// all-zero columns) a tiny ridge term, escalating until the system is
/// solvable, picks the minimum-norm-like solution instead of failing.
///
/// # Errors
///
/// `OlsError` when the input shape is invalid.
pub fn ols(rows: &[Vec<f64>], targets: &[f64]) -> Result<Vec<f64>, OlsError> {
    let n = rows.len();
    if n == 0 || n != targets.len() || rows[0].is_empty() {
        return Err(OlsError::BadShape);
    }
    let p = rows[0].len();
    if rows.iter().any(|r| r.len() != p) {
        return Err(OlsError::RaggedRows);
    }
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for (row, &y) in rows.iter().zip(targets) {
        for i in 0..p {
            xty[i] += row[i] * y;
            for j in 0..p {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    if let Some(beta) = solve(xtx.clone(), xty.clone()) {
        return Ok(beta);
    }
    #[allow(clippy::cast_precision_loss)] // p is a dozen
    let mean_diag = (0..p).map(|i| xtx[i][i]).sum::<f64>() / p as f64;
    let mut lambda = 1e-10 * mean_diag.max(1.0);
    loop {
        let mut ridged = xtx.clone();
        for (i, row) in ridged.iter_mut().enumerate() {
            row[i] += lambda;
        }
        if let Some(beta) = solve(ridged, xty.clone()) {
            return Ok(beta);
        }
        lambda *= 100.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-9, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn solve_needs_pivoting() {
        // First pivot is zero: 0x + 2y = 4, 3x + y = 5 -> x = 1, y = 2.
        let x = solve(vec![vec![0.0, 2.0], vec![3.0, 1.0]], vec![4.0, 5.0]).unwrap();
        close(&x, &[1.0, 2.0]);
    }

    #[test]
    fn solve_rejects_singular_systems() {
        assert!(solve(vec![vec![1.0, 2.0], vec![2.0, 4.0]], vec![1.0, 2.0]).is_none());
    }

    #[test]
    fn simple_line_fit_matches_hand_computation() {
        // Points (1,2) (2,3) (3,5): slope 1.5, intercept 1/3.
        let rows = vec![vec![1.0, 1.0], vec![1.0, 2.0], vec![1.0, 3.0]];
        let beta = ols(&rows, &[2.0, 3.0, 5.0]).unwrap();
        close(&beta, &[1.0 / 3.0, 1.5]);
    }

    #[test]
    fn exact_multivariate_relationship_is_recovered() {
        // y = 2 + 3 a - 1 b over a non-degenerate design.
        let pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 3.0), (4.0, 1.0)];
        let rows: Vec<Vec<f64>> = pts.iter().map(|&(a, b)| vec![1.0, a, b]).collect();
        let y: Vec<f64> = pts.iter().map(|&(a, b)| 2.0 + 3.0 * a - b).collect();
        close(&ols(&rows, &y).unwrap(), &[2.0, 3.0, -1.0]);
    }

    #[test]
    fn collinear_columns_fall_back_to_ridge_and_still_fit() {
        // Second and third columns identical: any split of the slope works.
        let rows: Vec<Vec<f64>> = (0..5).map(|i| vec![f64::from(i), f64::from(i)]).collect();
        let y: Vec<f64> = (0..5).map(|i| 4.0 * f64::from(i)).collect();
        let beta = ols(&rows, &y).unwrap();
        assert!((beta[0] + beta[1] - 4.0).abs() < 1e-6, "{beta:?}");
    }

    #[test]
    fn zero_column_gets_zero_coefficient() {
        let rows = vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0]];
        let beta = ols(&rows, &[2.0, 4.0, 6.0]).unwrap();
        close(&beta, &[2.0, 0.0]);
    }

    #[test]
    fn bad_shapes_are_errors() {
        assert_eq!(ols(&[], &[]), Err(OlsError::BadShape));
        assert_eq!(ols(&[vec![1.0]], &[1.0, 2.0]), Err(OlsError::BadShape));
        assert_eq!(
            ols(&[vec![1.0], vec![1.0, 2.0]], &[1.0, 2.0]),
            Err(OlsError::RaggedRows)
        );
    }
}
