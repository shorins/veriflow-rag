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
  groundedAnswer: string | null;
  faultInjectionActive: boolean;
  demoFaultMode: DraftMessage["demoFaultMode"];
  demoFaultCount: number;
  faultInjectionSummary: string | null;
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
    groundedAnswer: input.groundedAnswer,
    faultInjectionActive: input.faultInjectionActive,
    demoFaultMode: input.demoFaultMode,
    demoFaultCount: input.demoFaultCount,
    faultInjectionSummary: input.faultInjectionSummary,
    claims: [],
    highlightedSpans: [],
    activeClaimId: null,
    activeClaimSpan: null,
    activeRewrite: null,
    rewriteQueue: [],
    pendingVerificationCompleted: null,
    verificationState: "idle",
    verificationError: null,
  };
}

export function applyVerificationEvent(message: DraftMessage, event: VerificationEvent): DraftMessage {
  const payload = event.payload as Record<string, unknown>;

  if (event.event_type === "verification_started") {
    return {
      ...message,
      verificationState: "running",
      verificationError: null,
      activeClaimId: null,
      activeClaimSpan: null,
      activeRewrite: null,
      rewriteQueue: [],
      pendingVerificationCompleted: null,
    };
  }

  if (event.event_type === "claim_started") {
    return {
      ...message,
      verificationState: "running",
      activeClaimId: String(payload.claim_id),
      activeClaimSpan: String(payload.source_span),
      highlightedSpans: message.highlightedSpans.map((item) => ({
        ...item,
        isActive: item.claimId === String(payload.claim_id),
      })),
    };
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
            reason: claim.reason,
            revisedText: claim.revised_claim,
            isActive: message.activeClaimId === claim.claim_id,
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

  if (event.event_type === "rewrite_span_typing") {
    const newTask = {
      claimId: String(payload.claim_id),
      oldSpan: String(payload.old_span),
      newSpan: String(payload.new_span),
      diffSegments: ((payload.diff_segments as Array<{ kind: "equal" | "insert" | "delete"; value: string }>) ?? []),
      phase: "typing" as const,
    };

    if (!message.activeRewrite) {
      return {
        ...message,
        activeRewrite: newTask,
        verificationState: "rewriting",
      };
    } else {
      return {
        ...message,
        rewriteQueue: [...message.rewriteQueue, newTask],
        verificationState: "rewriting",
      };
    }
  }

  if (event.event_type === "verification_completed") {
    const finalAnswer = typeof payload.final_answer === "string" ? payload.final_answer : message.finalText;
    const rewrittenClaimIds = (((payload.applied_rewrites as Array<{ claim_id?: string }>) ?? [])
      .map((item) => item.claim_id)
      .filter(Boolean)) as string[];

    if (message.activeRewrite) {
      return {
        ...message,
        pendingVerificationCompleted: {
          finalAnswer,
          rewrittenClaimIds,
        },
      };
    } else {
      const rewrittenSet = new Set(rewrittenClaimIds);
      return {
        ...message,
        finalText: finalAnswer,
        displayText: finalAnswer,
        highlightedSpans: message.highlightedSpans
          .filter((item) => !rewrittenSet.has(item.claimId))
          .map((item) => ({ ...item, isActive: false })),
        activeClaimId: null,
        activeClaimSpan: null,
        activeRewrite: null,
        rewriteQueue: [],
        verificationState: "completed",
        pendingVerificationCompleted: null,
      };
    }
  }

  if (event.event_type === "run_error") {
    return {
      ...message,
      verificationState: "error",
      activeClaimId: null,
      activeClaimSpan: null,
      activeRewrite: null,
      rewriteQueue: [],
      pendingVerificationCompleted: null,
      verificationError: String(payload.message ?? "Verification run failed."),
    };
  }

  return message;
}

export function completeActiveRewrite(message: DraftMessage): DraftMessage {
  const completedRewrite = message.activeRewrite;
  if (!completedRewrite) return message;

  const newFinalText = message.finalText.replace(completedRewrite.oldSpan, completedRewrite.newSpan);
  const nextHighlights = message.highlightedSpans.filter((item) => item.claimId !== completedRewrite.claimId);
  const nextRewriteQueue = [...message.rewriteQueue];
  const nextActiveRewrite = nextRewriteQueue.shift() || null;

  if (!nextActiveRewrite && message.pendingVerificationCompleted) {
    const finalAnswer = message.pendingVerificationCompleted.finalAnswer;
    const rewrittenSet = new Set(message.pendingVerificationCompleted.rewrittenClaimIds);
    return {
      ...message,
      finalText: finalAnswer,
      displayText: finalAnswer,
      highlightedSpans: nextHighlights
        .filter((item) => !rewrittenSet.has(item.claimId))
        .map((item) => ({ ...item, isActive: false })),
      activeClaimId: null,
      activeClaimSpan: null,
      activeRewrite: null,
      rewriteQueue: [],
      verificationState: "completed",
      pendingVerificationCompleted: null,
    };
  }

  return {
    ...message,
    finalText: newFinalText,
    displayText: newFinalText,
    highlightedSpans: nextHighlights,
    activeRewrite: nextActiveRewrite,
    rewriteQueue: nextRewriteQueue,
  };
}
