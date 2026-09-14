"""
Shrike Optimization Algorithm (SOA) — a population-based, gradient-free search,
translated from the paper's Algorithm 1 / Eqs. 11-15 into working code.

Biological metaphor -> what it actually means numerically:
- "nest"          -> one candidate solution (a vector of numbers to search over)
- "parent bird"    -> a reference point used to generate a new candidate from an
                      existing one (either that nest's own best-so-far, or the
                      best found anywhere in the whole population)
- "feeding"        -> the update rule that nudges a candidate toward a reference
                      point, with a random step size and some extra randomness
- "no food"        -> if neither reference point produces an improvement, take
                      a random exploratory jump instead

This does NOT try to literally model birds/eggs/nests as separate objects (the
paper's own description of that structure is not precise enough to translate
1:1) — it's a faithful implementation of the paper's step-by-step search
*procedure* (Steps 1-6, Eqs. 11-15), same kind of judgment call as TSGF.
"""
import numpy as np


class ShrikeOptimizer:
    def __init__(self, dim: int, n_nests: int = 10, max_iter: int = 20,
                 bounds: tuple = (-1.0, 1.0), seed: int = 0):
        self.dim = dim
        self.n_nests = n_nests
        self.max_iter = max_iter
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)

    def optimize(self, fitness_fn, init_vector: np.ndarray = None, verbose: bool = True):
        """
        fitness_fn: callable(vector) -> float, HIGHER is better.
        init_vector: optional starting point (e.g. the already-trained weights)
                     — population is seeded as small perturbations around it,
                     so search starts near a known-good solution instead of
                     from scratch.
        Returns: (best_vector, best_fitness, fitness_history)
        """
        lo, hi = self.bounds

        # Step 1 & 2: Initialization + random generation
        if init_vector is not None:
            nests = init_vector[None, :] + self.rng.normal(0, 0.05, size=(self.n_nests, self.dim))
            nests = np.clip(nests, lo, hi)
        else:
            nests = self.rng.uniform(lo, hi, size=(self.n_nests, self.dim))

        # Step 3: Fitness function — evaluate every nest once
        fitness = np.array([fitness_fn(n) for n in nests])
        nest_best = nests.copy()               # each nest's own best-ever position ("male parent")
        nest_best_fitness = fitness.copy()

        global_best_idx = np.argmax(fitness)
        global_best = nests[global_best_idx].copy()   # best across the whole population ("female parent"/food source)
        global_best_fitness = fitness[global_best_idx]

        history = [global_best_fitness]

        for iteration in range(self.max_iter):
            for i in range(self.n_nests):
                s = self.rng.random()                    # natural factor in feeding (Eq. 13/14)
                alpha = self.rng.uniform(0, 2 * np.pi)    # constant factor / phase

                # Step 4: male-parent feeding (Eq. 13) — exploit toward a DIFFERENT,
                # randomly chosen nest's best-known spot (not this nest's own —
                # comparing a point against itself degenerates to a no-op the moment
                # it matches its own tracked best, which happens almost immediately
                # and was verified to cause premature convergence in testing below)
                other = self.rng.integers(0, self.n_nests)
                while other == i and self.n_nests > 1:
                    other = self.rng.integers(0, self.n_nests)
                candidate = s * (nests[i] - nest_best[other]) + nest_best[other]
                candidate = np.clip(candidate, lo, hi)
                cand_fitness = fitness_fn(candidate)

                if cand_fitness <= fitness[i]:
                    # Step 4 (Eq. 14) — didn't help; try female-parent feeding instead,
                    # exploiting the global best with an added sinusoidal exploration term
                    candidate = s * (nests[i] - global_best) + np.sin(alpha)
                    candidate = np.clip(candidate, lo, hi)
                    cand_fitness = fitness_fn(candidate)

                if cand_fitness <= fitness[i]:
                    # Step 5 (Eq. 15) — still no improvement ("no food"): random exploratory jump.
                    # A(n) (exploration strength) decays over iterations, matching the paper's
                    # general narrative of exploring more early, exploiting more later.
                    a_n = 1.0 - (iteration / max(self.max_iter, 1))
                    candidate = nests[i] + a_n * nests[i] + np.sin(alpha) * 0.1
                    candidate = np.clip(candidate, lo, hi)
                    cand_fitness = fitness_fn(candidate)

                if cand_fitness > fitness[i]:
                    nests[i] = candidate
                    fitness[i] = cand_fitness
                    if cand_fitness > nest_best_fitness[i]:
                        nest_best[i] = candidate
                        nest_best_fitness[i] = cand_fitness

            best_idx = np.argmax(fitness)
            if fitness[best_idx] > global_best_fitness:
                global_best = nests[best_idx].copy()
                global_best_fitness = fitness[best_idx]

            history.append(global_best_fitness)
            if verbose:
                print(f"[SOA] iter {iteration + 1}/{self.max_iter}  best_fitness={global_best_fitness:.4f}", flush=True)

        return global_best, global_best_fitness, history
