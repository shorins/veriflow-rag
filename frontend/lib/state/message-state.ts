import type { ClaimViewModel, DraftMessage, HighlightSpan, VerificationEvent } from "@/lib/types";

function claimStatusToHighlightStatus(status: ClaimViewModel["status"]): HighlightSpan["status"] | null {
  if (status === "supported") {
    return null;
  }
  return status;
}

export function createDraftMessage(input: {
  id: string;
  query: string;
  draftText: string;
  answerDepth: DraftMessage["answerDepth"];
  insufficientContext: boolean;
  citations: DraftMessage["citations"];
  draftModel: string;
  verificationModel: string;
  draftStrategy: DraftMessage["draftStrategy"];
  verificationSensitivity: DraftMessage["verificationSensitivity"];
}): DraftMessage {
  return {
    id: input.id,
    query: input.query,
    draftText: input.draftText,
    finalText: input.draftText,
    displayText: input.draftText,
    answerDepth: input.answerDepth,
    insufficientContext: input.insufficientContext,
    citations: input.citations,
    draftModel: input.draftModel,
    verificationModel: input.verificationModel,
    draftStrategy: input.draftStrategy,
    verificationSensitivity: input.verificationSensitivity,
    claims: [],
    highlightedSpans: [],
    activeRewrite: null,
    verificationState: "idle",
    verificationError: null,
  };
}

export function applyVerificationEvent(message: DraftMessage, event: VerificationEvent): DraftMessage {
  const payload = event.payload as Record<string, unknown>;

  if (event.event_type === "verification_started") {
    return { ...message, verificationState: "running", verificationError: null };
  }

  if (event.event_type === "claim_supported" || event.event_type === "claim_partial" || event.event_type === "claim_unsupported" || event.event_type === "claim_contradicted") {
    const claim: ClaimViewModel = {
      claim_id: String(payload.claim_id),
      claim_text: String(payload.claim_text),
      source_span: String(payload.source_span),
      status: payload.status as ClaimViewModel["status"],
      reason: String(payload.reason),
      used_evidence_ids: (payload.used_evidence_ids as string[]) ?? [],
      rewrite_needed: Boolean(payload.rewrite_needed),
      revised_claim: (payload.revised_claim as string | null) ?? null,
    };

    const highlightStatus = claimStatusToHighlightStatus(claim.status);
    const highlightedSpans = highlightStatus
      ? [
          ...message.highlightedSpans.filter((item) => item.claimId !== claim.claim_id),
          {
            claimId: claim.claim_id,
            sourceSpan: claim.source_span,
            status: highlightStatus,
            revisedText: claim.revised_claim,
          },
        ]
      : message.highlightedSpans;

    return {
      ...message,
      claims: [...message.claims.filter((item) => item.claim_id !== claim.claim_id), claim],
      highlightedSpans,
      activeRewrite:
        claim.rewrite_needed && claim.revised_claim
          ? {
              claimId: claim.claim_id,
              oldSpan: claim.source_span,
              newSpan: claim.revised_claim,
              diffSegments: [],
              phase: "erasing",
            }
          : message.activeRewrite,
      verificationState: claim.rewrite_needed ? "rewriting" : "running",
    };
  }

  if (event.event_type === "rewrite_span_erasing") {
    return {
      ...message,
      activeRewrite: {
        claimId: String(payload.claim_id),
        oldSpan: String(payload.old_span),
        newSpan: String(payload.old_span),
        diffSegments: [],
        phase: "erasing",
      },
      verificationState: "rewriting",
    };
  }

  if (event.event_type === "rewrite_span_typing") {
    return {
      ...message,
      activeRewrite: {
        claimId: String(payload.claim_id),
        oldSpan: String(payload.old_span),
        newSpan: String(payload.new_span),
        diffSegments: ((payload.diff_segments as Array<{ kind: "equal" | "insert" | "delete"; value: string }>) ?? []),
        phase: "typing",
      },
      verificationState: "rewriting",
    };
  }

  if (event.event_type === "rewrite_finished") {
    const oldSpan = String(payload.old_span);
    const newSpan = String(payload.new_span);
    return {
      ...message,
      finalText: message.finalText.replace(oldSpan, newSpan),
      displayText: message.finalText.replace(oldSpan, newSpan),
      highlightedSpans: message.highlightedSpans.map((item) =>
        item.claimId === String(payload.claim_id) ? { ...item, sourceSpan: newSpan, revisedText: newSpan } : item,
      ),
      activeRewrite: {
        claimId: String(payload.claim_id),
        oldSpan,
        newSpan,
        diffSegments: ((payload.diff_segments as Array<{ kind: "equal" | "insert" | "delete"; value: string }>) ?? []),
        phase: "done",
      },
      verificationState: "rewriting",
    };
  }

  if (event.event_type === "verification_completed") {
    const finalAnswer = typeof payload.final_answer === "string" ? payload.final_answer : message.finalText;
    return {
      ...message,
      finalText: finalAnswer,
      displayText: finalAnswer,
      activeRewrite: null,
      verificationState: "completed",
    };
  }

  if (event.event_type === "run_error") {
    return {
      ...message,
      verificationState: "error",
      activeRewrite: null,
      verificationError: String(payload.message ?? "Verification run failed."),
    };
  }

  return message;
}
