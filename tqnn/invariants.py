"""
Topological invariants for TQNN spin-network evaluation.

Implements the Jones polynomial V_L(t) via the Kauffman bracket and
Temperley-Lieb algebra. A braid word (list of signed integers) defines
a link whose closure yields the invariant.

Theory (compact):
  The Kauffman bracket <L> is computed from a braid closure by
  resolving each crossing into A-smoothing and A^{-1}-smoothing,
  then weighting by the loop factor delta = -(A^2 + A^{-2}).
  The Jones polynomial follows from the writhe-normalized bracket:
      V_L(t) = (-A)^{-3w} <L>  evaluated at  A = t^{-1/4}.

References:
  Kauffman, L. H. "State Models and the Jones Polynomial." Topology 26 (1987).
  Jones, V. F. R. "A polynomial invariant for knots via von Neumann algebras."
      Bull. AMS 12 (1985).
"""
from __future__ import annotations

import sympy
from sympy import Symbol, Rational, simplify, expand

A = Symbol("A")
t = Symbol("t")


def bracketPolynomial(braidWord: list[int], nStrands: int) -> sympy.Expr:
    """Compute the Kauffman bracket <L> for the closure of a braid.

    Each crossing is resolved recursively into A- and A^{-1}-smoothings.
    The result is a Laurent polynomial in A.

    Args:
        braidWord: List of signed integers. +i means sigma_i, -i means sigma_i^{-1}.
        nStrands:  Number of braid strands (must be >= 2).

    Returns:
        Kauffman bracket as a sympy expression in A.
    """
    if nStrands < 2:
        raise ValueError("nStrands must be >= 2")

    delta = -(A**2 + A**(-2))

    # State-sum approach: iterate over all 2^n smoothing choices
    nCrossings = len(braidWord)
    if nCrossings == 0:
        # Unknot: bracket = 1 (single loop, but normalized)
        return sympy.Integer(1)

    total = sympy.Integer(0)

    for bits in range(1 << nCrossings):
        # For each crossing, choose A-smoothing (bit=0) or A^{-1}-smoothing (bit=1)

        # State-sum: resolve each crossing and track connected components
        # via union-find on segments (layer * nStrands + strand)
        nLayers = nCrossings + 1
        segParent: list[int] = list(range(nLayers * nStrands))

        def segFind(x: int) -> int:
            while segParent[x] != x:
                segParent[x] = segParent[segParent[x]]
                x = segParent[x]
            return x

        def segUnion(x: int, y: int) -> None:
            px, py = segFind(x), segFind(y)
            if px != py:
                segParent[px] = py

        weight = sympy.Integer(1)

        for cIdx in range(nCrossings):
            crossing = braidWord[cIdx]
            sign = 1 if crossing > 0 else -1
            i = abs(crossing) - 1  # 0-indexed strand pair (i, i+1)

            if i < 0 or i >= nStrands - 1:
                raise ValueError(f"Crossing index {crossing} out of range for {nStrands} strands")

            bit = (bits >> cIdx) & 1

            topI = cIdx * nStrands + i
            topI1 = cIdx * nStrands + (i + 1)
            botI = (cIdx + 1) * nStrands + i
            botI1 = (cIdx + 1) * nStrands + (i + 1)

            if sign > 0:
                if bit == 0:
                    # A-smoothing of positive crossing: vertical pass-through
                    segUnion(topI, botI)
                    segUnion(topI1, botI1)
                    weight *= A
                else:
                    # A^{-1}-smoothing: horizontal reconnection
                    segUnion(topI, topI1)
                    segUnion(botI, botI1)
                    weight *= A**(-1)
            else:
                if bit == 0:
                    # A-smoothing of negative crossing: horizontal reconnection
                    segUnion(topI, topI1)
                    segUnion(botI, botI1)
                    weight *= A
                else:
                    # A^{-1}-smoothing: vertical pass-through
                    segUnion(topI, botI)
                    segUnion(topI1, botI1)
                    weight *= A**(-1)

            # All other strands pass through unchanged
            for s in range(nStrands):
                if s != i and s != (i + 1):
                    segUnion(cIdx * nStrands + s, (cIdx + 1) * nStrands + s)

        # Close the braid: connect top layer to bottom layer
        for s in range(nStrands):
            segUnion(s, nCrossings * nStrands + s)

        # Count distinct loops
        roots = set()
        for idx in range(nLayers * nStrands):
            roots.add(segFind(idx))
        nLoops = len(roots)

        total += weight * delta**(nLoops - 1)

    return expand(total)


def _writhe(braidWord: list[int]) -> int:
    """Compute the writhe (sum of crossing signs) of a braid word."""
    return sum(1 if c > 0 else -1 for c in braidWord)


def jonesPolynomial(braidWord: list[int], nStrands: int) -> sympy.Expr:
    """Compute the Jones polynomial V_L(t) for the closure of a braid.

    The Jones polynomial is obtained from the Kauffman bracket via
    writhe normalization:
        V_L(t) = (-A)^{-3w} <L>  with  A = t^{-1/4}

    Args:
        braidWord: List of signed integers encoding the braid.
        nStrands:  Number of braid strands.

    Returns:
        Jones polynomial as a sympy expression in t.
    """
    bracket = bracketPolynomial(braidWord, nStrands)
    w = _writhe(braidWord)
    normalizedBracket = expand((-A)**(-3 * w) * bracket)
    # Substitute A = t^{-1/4}
    jonesExpr = normalizedBracket.subs(A, t**Rational(-1, 4))
    return simplify(jonesExpr)


def verifyKnownValues() -> dict[str, bool]:
    """Verify the Jones polynomial against known values for standard knots.

    Returns:
        Dictionary mapping knot names to True/False for whether the
        computed value matches the known analytic result.
    """
    results: dict[str, bool] = {}

    # Unknot: trivial braid, V(t) = 1
    unknotJones = jonesPolynomial([], 2)
    results["unknot"] = simplify(unknotJones - 1) == 0

    # Trefoil (left-handed): braid word [1,1,1] on 2 strands
    # V(t) = -t^4 + t^3 + t
    trefoilJones = jonesPolynomial([1, 1, 1], 2)
    trefoilExpected = -t**4 + t**3 + t
    results["trefoil"] = simplify(trefoilJones - trefoilExpected) == 0

    # Figure-eight knot: braid word [1, -2, 1, -2] on 3 strands
    # V(t) = t^2 - t + 1 - t^{-1} + t^{-2}
    figEightJones = jonesPolynomial([1, -2, 1, -2], 3)
    figEightExpected = t**2 - t + 1 - t**(-1) + t**(-2)
    results["figure_eight"] = simplify(figEightJones - figEightExpected) == 0

    return results
