import math
from typing import Tuple, List


def _compute_score(br: int, bc: int, prefer_square: bool, prefer_larger: bool) -> float:
    """Internal scoring helper – no self needed"""
    score = br + bc
    if prefer_square and abs(br - bc) <= max(br, bc) * 0.2:
        score += 40
    if prefer_larger:
        score += (br * bc) ** 0.5 * 1.5   # reward larger blocks for very sparse matrices
    return score


def recommend_bsr_blocksize_any_shape(
    n_rows: int,
    n_cols: int,
    very_sparse: bool = True,
    max_block: int = 128,
    min_block: int = 4,
    prefer_square: bool = True,
    padding_tolerance: float = 0.05,
    prefer_larger: bool = True
) -> Tuple[Tuple[int, int], str]:
    """
    Recommend BSR blocksize for arbitrary matrix shape.
    Returns ((br, bc), explanation) or ((0,0), fallback suggestion)
    """
    def get_divisors(n: int) -> List[int]:
        divs = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted([d for d in divs if min_block <= d <= max_block], reverse=True)

    row_divs = get_divisors(n_rows)
    col_divs = get_divisors(n_cols)

    best_score = -1.0
    best_br, best_bc = 0, 0
    reasons = []

    # 1. Perfect divisors first
    for br in row_divs:
        for bc in col_divs:
            if n_rows % br == 0 and n_cols % bc == 0:
                score = _compute_score(br, bc, prefer_square, prefer_larger)
                if score > best_score:
                    best_score = score
                    best_br, best_bc = br, bc
                    reasons = [f"Perfect divisibility: {n_rows} % {br} == 0, {n_cols} % {bc} == 0"]

    # 2. Light padding if allowed (especially useful for very sparse)
    if padding_tolerance > 0 and (best_br == 0 or very_sparse):
        for br_target in range(max_block, min_block - 1, -1):
            for bc_target in range(max_block, min_block - 1, -1):
                pad_r = math.ceil(n_rows / br_target) * br_target - n_rows
                pad_c = math.ceil(n_cols / bc_target) * bc_target - n_cols

                rel_pad_r = pad_r / n_rows if n_rows > 0 else 0
                rel_pad_c = pad_c / n_cols if n_cols > 0 else 0
                rel_pad = max(rel_pad_r, rel_pad_c)

                if rel_pad <= padding_tolerance:
                    score = _compute_score(br_target, bc_target, prefer_square, prefer_larger)
                    score -= rel_pad * 50  # mild penalty for padding
                    if score > best_score:
                        best_score = score
                        best_br, best_bc = br_target, bc_target
                        reasons = [
                            f"Light padding: rows +{pad_r} ({rel_pad_r:.1%}), cols +{pad_c} ({rel_pad_c:.1%})",
                            f"padding ratio ≤ {padding_tolerance:.0%}"
                        ]

    if best_br > 0 and best_bc > 0:
        reason = "; ".join(reasons) if reasons else "Found suitable blocksize"
        return (best_br, best_bc), reason

    fallback = (
        f"No suitable blocksize found (min={min_block}, max={max_block}, padding≤{padding_tolerance:.0%}). "
        "Recommendation: use CSR format, or increase padding_tolerance / pad manually to multiple of 32/64."
    )
    return (0, 0), fallback


# Test call (this is the line that was failing)
block, reason = recommend_bsr_blocksize_any_shape(19251, 19251, very_sparse=True)
print("Recommended blocksize:", block)
print("Reason:", reason)