use crate::term::Term;

/// The standard SKI basis.
pub fn ski_basis() -> Vec<Term> {
    vec![Term::S, Term::K, Term::I]
}

/// Generate all CL terms of exactly `size` leaves, using the given basis
/// as the set of possible leaves.
///
/// A term of size N is a full binary tree with N leaves.
/// Count = Catalan(N-1) * |basis|^N
pub fn enumerate_size(size: usize, basis: &[Term]) -> Vec<Term> {
    if size == 0 {
        return vec![];
    }
    if size == 1 {
        return basis.to_vec();
    }
    let mut result = Vec::new();
    for left_size in 1..size {
        let right_size = size - left_size;
        let lefts = enumerate_size(left_size, basis);
        let rights = enumerate_size(right_size, basis);
        for l in &lefts {
            for r in &rights {
                result.push(Term::app(l.clone(), r.clone()));
            }
        }
    }
    result
}

/// Generate all terms for sizes 1..=max_size, using memoization.
/// Returns a Vec indexed by size (index 0 is empty).
pub fn enumerate_up_to(max_size: usize, basis: &[Term]) -> Vec<Vec<Term>> {
    let mut by_size: Vec<Vec<Term>> = vec![Vec::new(); max_size + 1];
    if max_size >= 1 {
        by_size[1] = basis.to_vec();
    }
    for n in 2..=max_size {
        let mut terms = Vec::new();
        for left_size in 1..n {
            let right_size = n - left_size;
            for l in &by_size[left_size] {
                for r in &by_size[right_size] {
                    terms.push(Term::app(l.clone(), r.clone()));
                }
            }
        }
        by_size[n] = terms;
    }
    by_size
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalan(n: usize) -> usize {
        if n <= 1 { return 1; }
        let mut c = 1u64;
        for i in 0..n as u64 {
            c = c * 2 * (2 * i + 1) / (i + 2);
        }
        c as usize
    }

    #[test]
    fn test_counts() {
        let basis = ski_basis();
        for n in 1..=6 {
            let terms = enumerate_size(n, &basis);
            let expected = catalan(n - 1) * 3usize.pow(n as u32);
            assert_eq!(
                terms.len(),
                expected,
                "size {n}: got {} expected {expected}",
                terms.len()
            );
        }
    }

    #[test]
    fn test_memoized_matches_direct() {
        let basis = ski_basis();
        let by_size = enumerate_up_to(5, &basis);
        for n in 1..=5 {
            let direct = enumerate_size(n, &basis);
            assert_eq!(by_size[n].len(), direct.len());
            // Same terms in same order
            assert_eq!(by_size[n], direct);
        }
    }
}
