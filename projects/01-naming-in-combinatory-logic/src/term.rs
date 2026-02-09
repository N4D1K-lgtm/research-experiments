use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// A term in combinatory logic, extensible with library combinators.
///
/// Uses Arc for O(1) cloning (critical for reduction performance) and
/// caches size in App nodes for O(1) size checks.
#[derive(Clone, Debug)]
pub enum Term {
    S,
    K,
    I,
    /// A user-defined combinator (index into a Library).
    Lib(usize),
    /// Application: (f x), with cached subtree size.
    App(Arc<Term>, Arc<Term>, usize),
}

impl PartialEq for Term {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Term::S, Term::S) | (Term::K, Term::K) | (Term::I, Term::I) => true,
            (Term::Lib(a), Term::Lib(b)) => a == b,
            (Term::App(f1, x1, _), Term::App(f2, x2, _)) => f1 == f2 && x1 == x2,
            _ => false,
        }
    }
}

impl Eq for Term {}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Term::S | Term::K | Term::I => {}
            Term::Lib(i) => i.hash(state),
            Term::App(f, x, _) => {
                f.hash(state);
                x.hash(state);
            }
        }
    }
}

impl Term {
    pub fn app(f: Term, x: Term) -> Term {
        let size = f.size() + x.size();
        Term::App(Arc::new(f), Arc::new(x), size)
    }

    /// Number of combinator leaves. O(1) — cached for App nodes.
    pub fn size(&self) -> usize {
        match self {
            Term::S | Term::K | Term::I | Term::Lib(_) => 1,
            Term::App(_, _, size) => *size,
        }
    }

    /// Tree depth (0 for atoms).
    pub fn depth(&self) -> usize {
        match self {
            Term::S | Term::K | Term::I | Term::Lib(_) => 0,
            Term::App(f, x, _) => 1 + f.depth().max(x.depth()),
        }
    }

    /// Decompose into application spine: (head, [arg0, arg1, ...])
    /// e.g. ((S K) I) -> (S, [K, I])
    pub fn spine(&self) -> (&Term, Vec<&Term>) {
        let mut head = self;
        let mut args = Vec::new();
        while let Term::App(f, x, _) = head {
            args.push(x.as_ref());
            head = f.as_ref();
        }
        args.reverse();
        (head, args)
    }

    /// All sub-terms including self, depth-first.
    pub fn subterms(&self) -> Vec<&Term> {
        let mut result = vec![self];
        if let Term::App(f, x, _) = self {
            result.extend(f.subterms());
            result.extend(x.subterms());
        }
        result
    }

    /// Is this an atom (not an application)?
    pub fn is_atom(&self) -> bool {
        !matches!(self, Term::App(_, _, _))
    }

    /// Replace all Lib(i) nodes with their CL expansions.
    /// Used to normalize terms back to pure SKI for cross-basis comparison.
    pub fn expand_libs(&self, expansions: &[Term]) -> Term {
        match self {
            Term::S | Term::K | Term::I => self.clone(),
            Term::Lib(i) => expansions.get(*i).cloned().unwrap_or_else(|| self.clone()),
            Term::App(f, x, _) => Term::app(f.expand_libs(expansions), x.expand_libs(expansions)),
        }
    }
}

/// Left-associative display: SK means (S K), SKI means ((S K) I),
/// S(KI) means (S (K I)). Parens only on the right side of an App
/// when the right side is also an App.
impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::S => write!(f, "S"),
            Term::K => write!(f, "K"),
            Term::I => write!(f, "I"),
            Term::Lib(i) => write!(f, "#{i}"),
            Term::App(func, arg, _) => {
                write!(f, "{func}")?;
                match arg.as_ref() {
                    Term::App(_, _, _) => write!(f, "({arg})"),
                    _ => write!(f, "{arg}"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        assert_eq!(Term::S.size(), 1);
        assert_eq!(Term::app(Term::S, Term::K).size(), 2);
        assert_eq!(Term::app(Term::app(Term::S, Term::K), Term::I).size(), 3);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Term::app(Term::S, Term::K)), "SK");
        assert_eq!(
            format!("{}", Term::app(Term::app(Term::S, Term::K), Term::I)),
            "SKI"
        );
        assert_eq!(
            format!("{}", Term::app(Term::S, Term::app(Term::K, Term::I))),
            "S(KI)"
        );
    }

    #[test]
    fn test_spine() {
        let ski = Term::app(Term::app(Term::S, Term::K), Term::I);
        let (head, args) = ski.spine();
        assert_eq!(*head, Term::S);
        assert_eq!(args.len(), 2);
        assert_eq!(*args[0], Term::K);
        assert_eq!(*args[1], Term::I);
    }
}
