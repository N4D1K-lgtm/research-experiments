# Strange Loops: Research Questions & Experimentation Plans

## Core Thesis

Structure emerges from self-referential process. Recursion generates abstraction.
Language structures thought. Articulation structures communication. Spectra structure
knowledge. Human cognition is efficient because it operates on all of these
simultaneously - and current AI operates on almost none of them.

---

## Cluster A: Recursion & Projection (Ideas 1, 5)

### A1: What is the minimum recursive depth required for abstraction?

**Observation:** A function that can call itself can represent unbounded computation.
But abstraction - the ability to treat a *pattern* as a *thing* - seems to require
something more specific: the ability to apply an operation to its own output and
recognize the resulting structure.

**Research question:** In a formal system where the only power is recursive
self-application, at what depth do novel structural patterns emerge that weren't
explicit in the base rules? Is there a phase transition?

**Experiment sketch:**
- Define a minimal rewriting system (lambda calculus, combinatory logic, or
  something simpler)
- Allow it to apply rules to its own outputs recursively
- At each depth, measure structural complexity (Kolmogorov complexity estimate,
  graph structure of terms, emergence of repeated sub-patterns)
- Look for a critical depth where complexity jumps - a "phase transition" into
  abstraction
- Compare: does this depth change with different base rule sets?

**Key metric:** When does the system start producing structures that compress
better than their derivation? (i.e., when does the *output* become more useful
as a building block than the *process* that created it?)

---

### A2: Is projection just recursion with a world model?

**Observation:** Humans simulate outcomes before acting. This "projection" seems
like recursion applied not to formal symbols but to an internal model of the
world. The question is: what's the minimum world model that makes projection
useful?

**Research question:** Given a task environment, what is the minimum fidelity
internal model that, combined with recursive simulation, outperforms direct
reactive behavior? How does this minimum fidelity scale with task complexity?

**Experiment sketch:**
- Define a family of tasks with varying complexity (grid worlds, planning
  problems, constraint satisfaction)
- Build agents with internal models of varying fidelity (from "no model" through
  "crude sketch" to "perfect copy")
- Allow agents to "simulate" by running their model forward recursively before
  acting
- Measure: at what model fidelity does simulation start helping? Does this
  threshold depend on task structure?
- Hypothesis: there's a sweet spot where a *very coarse* model + recursion
  beats a detailed model without recursion

**Connection to A1:** The "phase transition" in A1 might correspond to the
point where the system's self-model becomes useful for projection.

---

## Cluster B: Language, Monologue & Learning Efficiency (Ideas 2, 3, 4)

### B1: Does self-narration accelerate learning?

**Observation:** Humans talk to themselves constantly. Children who narrate their
actions learn faster. Current AI has chain-of-thought, but it doesn't *learn from*
its own narration - the narration is disposable scaffolding.

**Research question:** If a learning system generates linguistic descriptions of
its experiences and then treats those descriptions as additional training data,
does it learn faster (i.e., reach equivalent performance on fewer examples)?

**Experiment sketch:**
- Task: any standard few-shot learning benchmark
- Agent A: trains on raw examples (input -> output)
- Agent B: trains on raw examples AND its own generated descriptions of those
  examples ("I saw X, and I think the pattern is Y, because Z")
- Agent C: trains ONLY on its own descriptions (no raw examples after initial
  exposure)
- Measure: learning curves for A, B, C
- Hypothesis: B > A (narration helps), and the gap widens as data gets scarcer
- Key question: does Agent C eventually match A? That would mean language
  is sufficient for transferring the learning signal.

**Deeper question:** Is the narration valuable because it's *compressed*? Or
because it's *structured*? Design an ablation: compare natural language
narration vs. random-order feature lists vs. compressed binary encoding.

---

### B2: What properties must a language have to be useful for thought?

**Observation:** Not all symbol systems are equally useful for thinking. Natural
languages have specific properties (compositionality, recursion, ambiguity,
graded meaning) that seem important. Which of these actually matter for
cognitive efficiency?

**Research question:** If you let a communication protocol evolve under joint
pressure for (a) expressiveness, (b) compression, and (c) learnability, which
structural properties emerge? Are they the same ones natural languages have?

**Experiment sketch:**
- Emergent communication game: two agents must communicate about a structured
  world through a bandwidth-limited channel
- Evolutionary pressure: agents that communicate better survive/reproduce
- Add a twist: agents must also be able to learn the protocol quickly (new
  agents are periodically introduced and must learn from few examples)
- Track: does the protocol develop compositional structure? Hierarchy? Zipfian
  frequency distribution? Ambiguity? Redundancy?
- Compare protocols optimized for expressiveness-only vs.
  expressiveness+learnability
- Hypothesis: the learnability pressure is what forces compositionality and
  hierarchy to emerge (because these make the system learnable from few examples)

**Connection to B1:** If the evolved protocol has the "right" properties, does
using it for self-narration (B1) work better than using an arbitrary symbol
system?

---

### B3: Why do humans learn faster than AI on less data?

**Observation:** A child can learn a concept from 1-3 examples. GPT-4 needs
thousands-to-millions. The standard explanation is "prior knowledge" or
"inductive bias," but what if the real answer is that humans have a better
*encoding*?

**Research question:** Is the key advantage of human learning the ability to
compress experience into *narratives* (temporally structured, causally linked
descriptions) rather than *features* (static, independent attributes)?

**Experiment sketch:**
- Take a concept learning task
- Represent training examples three ways:
  (a) Feature vectors (standard ML)
  (b) Descriptive text (language model style)
  (c) Narratives with causal structure ("first I noticed X, which made me
      think Y, and then when I saw Z it confirmed...")
- Train models on each representation
- Measure sample efficiency
- Hypothesis: narrative representations lead to dramatically better sample
  efficiency, because they encode not just WHAT but WHY and HOW

**Provocative sub-question:** What if the reason LLMs need so much data is
precisely because they process language as flat token sequences rather than
as structured narratives with causal/temporal backbone?

---

## Cluster C: Articulation & Spectra (Ideas 6, 7)

### C1: Can relative articulation outperform categorical symbols?

**Observation:** In human speech, phonemes aren't fixed symbols - they're
*gestures* defined by relative positions of articulators (tongue height, lip
rounding, voicing). Meaning comes from *contrasts between positions*, not
from the positions themselves. This is radically different from how we
typically encode information in computing (as discrete categorical symbols).

**Research question:** Is a communication system based on continuous
"articulatory gestures" (where meaning is encoded in relative positions
along continuous dimensions) more efficient than one based on discrete
categorical symbols? Under what conditions?

**Experiment sketch:**
- Design two communication protocols:
  (a) Categorical: fixed alphabet of discrete symbols
  (b) Articulatory: messages are points in a continuous space, meaning is
      determined by relative position to learned "reference points"
- Same communication game, same bandwidth constraints
- Measure: expressiveness, compressibility, learnability, robustness to noise
- Hypothesis: the articulatory system is more robust to noise (because small
  perturbations don't cross category boundaries in the same way) and more
  expressive per bit (because it can encode gradient meaning)

**Key insight to test:** In the articulatory system, adding a new concept
doesn't require adding a new symbol - it requires adding a new *direction*
in the existing space. This should scale better.

---

### C2: Is spectrum reasoning a better knowledge representation?

**Observation:** Humans rarely think in absolutes. "Is this hot?" gets
processed as "how hot, relative to what?" Knowledge is stored as positions
on spectra with named extremes. "Steel is hard" means "steel is closer to
the hard end of the hard-soft spectrum."

**Research question:** Can a knowledge representation based on continuous
spectra (where facts are positions between named extremes) support better
reasoning than categorical (true/false) or probabilistic (0-1) representations?

**Experiment sketch:**
- Represent a domain of knowledge three ways:
  (a) Propositional: "steel is hard" (true/false)
  (b) Probabilistic: "P(steel is hard) = 0.95"
  (c) Spectral: "steel is at position 0.92 on the [soft...hard] spectrum,
      0.85 on the [flexible...rigid] spectrum, 0.3 on the [light...heavy]
      spectrum"
- Test reasoning tasks: analogy, transfer, novel inference
- Example reasoning task: "If X is like steel but softer, what properties
  might X have?" The spectral representation directly supports this (move
  along the soft-hard axis); the propositional representation doesn't.
- Hypothesis: spectral representation enables analogical and transfer
  reasoning that the others struggle with

**Deep question:** Where do the spectra come from? Hypothesis: they emerge
from experience (you learn what "hot" and "cold" mean, and then everything
thermal gets placed on that spectrum). Can we make spectra emerge from data
rather than defining them a priori?

---

## Cross-Cutting Questions

### X1: Do these mechanisms reinforce each other?

**Hypothesis:** The real power isn't in any single mechanism but in their
combination. Specifically:
- Recursion (A) generates the *capacity* for abstraction
- Language (B) provides the *medium* for abstraction
- Articulation (C) provides the *encoding* for abstraction
- Together: recursive self-narration using an articulatory encoding over
  spectral representations

**Experiment:** Build a system that has all three and compare it to systems
that have any two. Is there a synergy?

### X2: What's the minimal "cognitive architecture"?

**Question:** What is the smallest system that exhibits all of these
properties simultaneously? Can we find it by starting from a minimal
substrate and adding capabilities one at a time, measuring when each new
capability produces a qualitative jump?

### X3: Consciousness as the interaction effect?

**Speculation (not testable yet, but worth formalizing):** Consciousness
might be what happens when a recursive system models itself modeling itself,
using a language-like encoding, with spectral (graded, not binary) state.
The "hard problem" might dissolve if we realize that the *experience* of
consciousness is just what recursive self-modeling *is* from the inside.

Can we at least formalize what "recursive self-modeling" means precisely
enough to reason about it mathematically?

---

## Prioritized Starting Points

1. **A1 (Recursion phase transition)** - Most formally tractable. Can be
   done with pure math/computation, no ML needed. Results inform everything
   else.

2. **C2 (Spectrum reasoning)** - Relatively self-contained. Could produce
   a novel knowledge representation that's immediately useful.

3. **B1 (Self-narration)** - Directly testable against existing ML
   benchmarks. High potential for a publishable result.

4. **C1 (Articulatory encoding)** - Novel and under-explored in the
   literature. Connects to information theory in interesting ways.

5. **B2 (Language emergence)** - Builds on existing emergent communication
   literature but adds the learnability pressure angle.
