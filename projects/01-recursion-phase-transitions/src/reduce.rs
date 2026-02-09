use crate::term::Term;

// ---------------------------------------------------------------------------
// Library: extensible combinator definitions (for the naming experiment)
// ---------------------------------------------------------------------------

/// A template for a combinator body. Arg(n) refers to the nth argument.
#[derive(Clone, Debug)]
pub enum Template {
    Arg(usize),
    S,
    K,
    I,
    Lib(usize),
    App(Box<Template>, Box<Template>),
}

impl Template {
    pub fn app(f: Template, x: Template) -> Template {
        Template::App(Box::new(f), Box::new(x))
    }

    /// Substitute arguments into the template, producing a concrete Term.
    pub fn instantiate(&self, args: &[Term]) -> Term {
        match self {
            Template::Arg(n) => args[*n].clone(),
            Template::S => Term::S,
            Template::K => Term::K,
            Template::I => Term::I,
            Template::Lib(i) => Term::Lib(*i),
            Template::App(f, x) => {
                Term::app(f.instantiate(args), x.instantiate(args))
            }
        }
    }
}

pub struct LibEntry {
    pub name: String,
    pub arity: usize,
    pub body: Template,
}

pub struct Library {
    pub entries: Vec<LibEntry>,
}

impl Library {
    pub fn new() -> Self {
        Library { entries: Vec::new() }
    }

    /// Add a combinator with an explicit template. Returns its index.
    pub fn add(&mut self, name: impl Into<String>, arity: usize, body: Template) -> usize {
        let idx = self.entries.len();
        self.entries.push(LibEntry {
            name: name.into(),
            arity,
            body,
        });
        idx
    }

    /// Add a named combinator from a CL term (e.g. a discovered motif).
    /// Computes effective arity from the term's spine and builds the
    /// expansion template automatically.
    pub fn add_term(&mut self, name: impl Into<String>, term: &Term) -> usize {
        let arity = effective_arity(term);
        // Body = term applied to Arg(0)..Arg(arity-1)
        let mut body = Template::from_term(term);
        for i in 0..arity {
            body = Template::app(body, Template::Arg(i));
        }
        self.add(name, arity, body)
    }

    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }
}

/// Compute how many additional args a term needs before its head combinator fires.
pub fn effective_arity(term: &Term) -> usize {
    let (head, args) = term.spine();
    let head_arity: usize = match head {
        Term::S => 3,
        Term::K => 2,
        Term::I => 1,
        _ => 0,
    };
    head_arity.saturating_sub(args.len())
}

impl Template {
    /// Convert a concrete Term into a Template (no Arg nodes).
    pub fn from_term(term: &Term) -> Template {
        match term {
            Term::S => Template::S,
            Term::K => Template::K,
            Term::I => Template::I,
            Term::Lib(i) => Template::Lib(*i),
            Term::App(f, x, _) => Template::app(
                Template::from_term(f),
                Template::from_term(x),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionStatus {
    NormalForm,
    OutOfFuel,
    TermTooLarge,
}

#[derive(Debug, Clone)]
pub struct ReductionResult {
    pub term: Term,
    pub steps: usize,
    pub status: ReductionStatus,
}

/// Try to reduce the outermost (leftmost) redex. Returns None if already
/// in normal form.
fn try_spine_reduce(term: &Term, lib: &Library) -> Option<Term> {
    let (head, args) = term.spine();

    let (result, rest_start) = match head {
        Term::I if !args.is_empty() => {
            (args[0].clone(), 1)
        }
        Term::K if args.len() >= 2 => {
            (args[0].clone(), 2)
        }
        Term::S if args.len() >= 3 => {
            let xz = Term::app(args[0].clone(), args[2].clone());
            let yz = Term::app(args[1].clone(), args[2].clone());
            (Term::app(xz, yz), 3)
        }
        Term::Lib(i) => {
            let entry = lib.entries.get(*i)?;
            if args.len() >= entry.arity {
                let owned: Vec<Term> = args[..entry.arity]
                    .iter()
                    .map(|a| (*a).clone())
                    .collect();
                (entry.body.instantiate(&owned), entry.arity)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    // Rebuild application with any remaining args
    let mut out = result;
    for arg in &args[rest_start..] {
        out = Term::app(out, (*arg).clone());
    }
    Some(out)
}

/// One normal-order reduction step.
fn step(term: &Term, lib: &Library) -> Option<Term> {
    // 1. Try outermost spine reduction
    if let Some(r) = try_spine_reduce(term, lib) {
        return Some(r);
    }

    // 2. No outermost redex — recurse into sub-terms (left then right)
    if let Term::App(f, x, _) = term {
        if let Some(f2) = step(f, lib) {
            return Some(Term::app(f2, x.as_ref().clone()));
        }
        if let Some(x2) = step(x, lib) {
            return Some(Term::app(f.as_ref().clone(), x2));
        }
    }

    None
}

/// Reduce a term to normal form (or until fuel/size limits are hit).
pub fn reduce(term: &Term, lib: &Library, fuel: usize, max_size: usize) -> ReductionResult {
    let mut current = term.clone();
    let mut steps = 0;

    loop {
        if steps >= fuel {
            return ReductionResult {
                term: current,
                steps,
                status: ReductionStatus::OutOfFuel,
            };
        }
        if current.size() > max_size {
            return ReductionResult {
                term: current,
                steps,
                status: ReductionStatus::TermTooLarge,
            };
        }
        match step(&current, lib) {
            Some(next) => {
                current = next;
                steps += 1;
            }
            None => {
                return ReductionResult {
                    term: current,
                    steps,
                    status: ReductionStatus::NormalForm,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lib() -> Library {
        Library::new()
    }

    fn nf(term: &Term) -> Term {
        reduce(term, &lib(), 1000, 10000).term
    }

    #[test]
    fn test_i_reduction() {
        // I S -> S
        let t = Term::app(Term::I, Term::S);
        assert_eq!(nf(&t), Term::S);
    }

    #[test]
    fn test_k_reduction() {
        // K S I -> S
        let t = Term::app(Term::app(Term::K, Term::S), Term::I);
        assert_eq!(nf(&t), Term::S);
    }

    #[test]
    fn test_s_reduction() {
        // S K I x -> x  (for any x; use S as x)
        let skis = Term::app(
            Term::app(Term::app(Term::S, Term::K), Term::I),
            Term::S,
        );
        assert_eq!(nf(&skis), Term::S);
    }

    #[test]
    fn test_skk_is_identity() {
        // SKK x -> x for any x
        for x in [Term::S, Term::K, Term::I] {
            let t = Term::app(
                Term::app(Term::app(Term::S, Term::K), Term::K),
                x.clone(),
            );
            assert_eq!(nf(&t), x);
        }
    }

    #[test]
    fn test_omega_diverges() {
        // SII(SII) -> divergent
        let sii = Term::app(Term::app(Term::S, Term::I), Term::I);
        let omega = Term::app(sii.clone(), sii);
        let result = reduce(&omega, &lib(), 100, 10000);
        assert_eq!(result.status, ReductionStatus::OutOfFuel);
    }

    #[test]
    fn test_inner_reduction() {
        // S(IK) -> SK (reduce inside arg when head doesn't have enough args)
        let t = Term::app(Term::S, Term::app(Term::I, Term::K));
        assert_eq!(nf(&t), Term::app(Term::S, Term::K));
    }

    #[test]
    fn test_library_combinator() {
        // B = S(KS)K, B x y z = x(yz)
        let mut lib = Library::new();
        let b_body = Template::app(
            Template::Arg(0),
            Template::app(Template::Arg(1), Template::Arg(2)),
        );
        let b_idx = lib.add("B", 3, b_body);

        // B K S I -> K(SI)
        let t = Term::app(
            Term::app(
                Term::app(Term::Lib(b_idx), Term::K),
                Term::S,
            ),
            Term::I,
        );
        let result = reduce(&t, &lib, 1000, 10000);
        // B K S I = K(SI) -> reduce K(SI) ... K needs 2 args, has 1. NF = K(SI)
        assert_eq!(result.term, Term::app(Term::K, Term::app(Term::S, Term::I)));
    }

    #[test]
    fn test_add_term_ss() {
        // SS = App(S, S), arity should be 2
        // SS x y = S S x y = S y (x y)
        let ss = Term::app(Term::S, Term::S);
        let mut lib = Library::new();
        let idx = lib.add_term("SS", &ss);
        assert_eq!(lib.entries[idx].arity, 2);

        // #0 K I should behave like SS K I = S I (K I) = S I (K I)
        // Actually: S S K I = SK(KI)... let me trace:
        // S S K I: S with args [S, K, I] -> S I (K I)
        // Then S I (K I) needs one more arg -> normal form
        let t = Term::app(
            Term::app(Term::Lib(idx), Term::K),
            Term::I,
        );
        let result = reduce(&t, &lib, 1000, 10000);

        // Direct: SS K I = S S K I
        let direct_t = Term::app(Term::app(Term::app(Term::S, Term::S), Term::K), Term::I);
        let empty = Library::new();
        let direct_result = reduce(&direct_t, &empty, 1000, 10000);

        assert_eq!(result.term, direct_result.term);
    }

    #[test]
    fn test_lib_stays_in_nf() {
        // Lib(0) by itself (no args) should be a normal form
        let ss = Term::app(Term::S, Term::S);
        let mut lib = Library::new();
        lib.add_term("SS", &ss);

        let t = Term::Lib(0);
        let result = reduce(&t, &lib, 1000, 10000);
        assert_eq!(result.status, ReductionStatus::NormalForm);
        assert_eq!(result.term, Term::Lib(0));
    }

    #[test]
    fn test_expand_libs() {
        let ss = Term::app(Term::S, Term::S);
        let t = Term::app(Term::Lib(0), Term::K);
        let expanded = t.expand_libs(&[ss]);
        assert_eq!(expanded, Term::app(Term::app(Term::S, Term::S), Term::K));
    }
}
