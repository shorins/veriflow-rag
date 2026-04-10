from __future__ import annotations

import html

from veriflow_rag.verification.claims import find_span_range
from veriflow_rag.verification.models import AppliedRewrite, ClaimVerificationResult


STATUS_COLORS = {
    "supported": "#d1fae5",
    "partial": "#fef3c7",
    "unsupported": "#fecaca",
    "contradicted": "#fca5a5",
}

STATUS_LABELS = {
    "supported": "supported",
    "partial": "partial",
    "unsupported": "unsupported",
    "contradicted": "contradicted",
}


def render_claim_table(claim_results: list[ClaimVerificationResult]) -> str:
    if not claim_results:
        return "### Claims\n\n_Claims не найдены._"

    lines = [
        "### Claims",
        "",
        "| Claim | Status | Reason | Evidence |",
        "|---|---|---|---|",
    ]
    for result in claim_results:
        evidence = ", ".join(result.used_evidence_ids) or "—"
        claim_text = result.claim_text.replace("|", "\\|")
        reason = result.reason.replace("|", "\\|")
        lines.append(
            f"| `{result.claim_id}` {claim_text} | `{result.status}` | {reason} | {evidence} |"
        )
    return "\n".join(lines)


def render_highlighted_answer(draft_answer: str, claim_results: list[ClaimVerificationResult]) -> str:
    if not claim_results:
        return f"<div>{html.escape(draft_answer)}</div>"

    ranges = []
    for result in claim_results:
        span_range = find_span_range(draft_answer, result.source_span)
        if span_range is None:
            continue
        ranges.append((span_range[0], span_range[1], result))

    ranges.sort(key=lambda item: item[0])
    pieces: list[str] = []
    cursor = 0
    for start, end, result in ranges:
        if start < cursor:
            continue
        pieces.append(html.escape(draft_answer[cursor:start]))
        text = html.escape(draft_answer[start:end])
        style = f"background:{STATUS_COLORS[result.status]};padding:2px 4px;border-radius:4px;"
        if result.status == "contradicted":
            style += "text-decoration:line-through;"
        pieces.append(
            f'<span style="{style}" title="{html.escape(result.reason)}">{text}</span>'
        )
        cursor = end
    pieces.append(html.escape(draft_answer[cursor:]))
    return "<div style='line-height:1.7'>" + "".join(pieces) + "</div>"


def render_rewrite_diff(applied_rewrites: list[AppliedRewrite]) -> str:
    if not applied_rewrites:
        return "### Rewrites\n\n_Локальные переписывания не применялись._"

    lines = ["### Rewrites", ""]
    for rewrite in applied_rewrites:
        lines.extend(
            [
                f"**{rewrite.claim_id}** (`{rewrite.status_before}`)",
                "",
                f"- Before: `{rewrite.old_span}`",
                f"- After: `{rewrite.new_span}`",
                "",
            ]
        )
    return "\n".join(lines)
