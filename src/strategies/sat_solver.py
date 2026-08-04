"""SAT encoding for Turyn TT(n) sequences via PySAT."""
from __future__ import annotations

import time
import numpy as np

try:
    from pysat.solvers import Glucose4
    from pysat.card import CardEnc, EncType
    _HAS_PYSAT = True
except ImportError:
    _HAS_PYSAT = False


class TurynSatEncoder:
    """Encode TT(n) as CNF and solve via SAT."""

    def __init__(self, n: int):
        if n < 2:
            raise ValueError("n >= 2 required")
        self.n = n
        self.lengths = (n, n, n, n - 1)
        self.weights = (1, 1, 2, 2)

    def solve(self, timeout_s: int = 300):
        if not _HAS_PYSAT:
            raise RuntimeError("python-sat not installed")
        return self._solve(timeout_s)

    def _solve(self, timeout_s):
        n = self.n
        lengths = self.lengths
        weights = self.weights

        next_id = 1
        x_id: dict[tuple[int, int], int] = {}
        for si, L in enumerate(lengths):
            for pos in range(L):
                x_id[(si, pos)] = next_id
                next_id += 1

        solver = Glucose4()
        clauses = []

        for shift in range(1, n):
            product_lits = []
            for si, L in enumerate(lengths):
                w = weights[si]
                if shift >= L:
                    continue
                for i in range(L - shift):
                    a = x_id[(si, i)]
                    b = x_id[(si, i + shift)]
                    p = next_id
                    next_id += 1
                    clauses.append([-p, -a, b])
                    clauses.append([-p, a, -b])
                    clauses.append([a, b, p])
                    clauses.append([-a, -b, p])
                    for _ in range(w):
                        product_lits.append(p)

            total = sum(
                weights[si] * max(0, lengths[si] - shift) for si in range(4))
            if total % 2 != 0:
                solver.delete()
                return None, False
            bound = total // 2

            if not product_lits:
                continue

            c_atmost = CardEnc.atmost(
                lits=product_lits, bound=bound,
                top_id=next_id, encoding=EncType.seqcounter)
            clauses.extend(c_atmost.clauses)
            if c_atmost.clauses:
                next_id = max(next_id, max(abs(l) for cl in c_atmost.clauses for l in cl))

            c_atleast = CardEnc.atmost(
                lits=[-p for p in product_lits], bound=total - bound,
                top_id=next_id + 1, encoding=EncType.seqcounter)
            clauses.extend(c_atleast.clauses)
            if c_atleast.clauses:
                next_id = max(next_id, max(abs(l) for cl in c_atleast.clauses for l in cl))

        for cl in clauses:
            solver.add_clause(cl)

        print(f"  SAT: {next_id} vars, {len(clauses)} clauses, solving...",
              end=" ", flush=True)
        started = time.perf_counter()
        status = solver.solve_limited(expect_interrupt=True)
        elapsed = time.perf_counter() - started
        print(f"{elapsed:.1f}s")

        if status:
            model = solver.get_model()
            seq = np.zeros((4, n), dtype=np.int8)
            for si, L in enumerate(lengths):
                for pos in range(L):
                    val = model[x_id[(si, pos)] - 1]
                    seq[si, pos] = 1 if val > 0 else -1
            solver.delete()
            return seq, True
        else:
            solver.delete()
            return None, False
