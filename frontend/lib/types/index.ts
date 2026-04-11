export type DraftStrategy = "conservative" | "balanced" | "demo";
export type VerificationSensitivity = "conservative" | "balanced" | "demo";
export type AnswerDepth = "brief" | "standard" | "detailed";
export type DocumentStatus = "uploaded" | "indexed" | "stale" | "error";
export type ClaimStatus = "supported" | "partial" | "unsupported" | "contradicted";

export type Citation = {
  evidence_id: string;
  file_name: string;
  section_title: string;
  support: string;
};

export type DocumentItem = {
  document_id: string;
  file_name: string;
  stored_path: string;
  size_bytes: number;
  uploaded_at: string;
  status: DocumentStatus;
  last_parsed_at: string | null;
  last_indexed_at: string | null;
  error_message: string | null;
};

export type ClaimViewModel = {
  claim_id: string;
  claim_text: string;
  source_span: string;
  status: ClaimStatus;
  reason: string;
  used_evidence_ids: string[];
  rewrite_needed: boolean;
  revised_claim: string | null;
};

export type HighlightSpan = {
  claimId: string;
  sourceSpan: string;
  status: Exclude<ClaimStatus, "supported">;
  revisedText?: string | null;
};

export type RewriteDiffSegment = {
  kind: "equal" | "insert" | "delete";
  value: string;
};

export type RewriteAnimationState = {
  claimId: string;
  oldSpan: string;
  newSpan: string;
  diffSegments: RewriteDiffSegment[];
  phase: "erasing" | "typing" | "done";
} | null;

export type VerificationState = "idle" | "running" | "rewriting" | "completed" | "error";

export type DraftMessage = {
  id: string;
  query: string;
  draftText: string;
  finalText: string;
  displayText: string;
  answerDepth: AnswerDepth;
  insufficientContext: boolean;
  citations: Citation[];
  draftModel: string;
  verificationModel: string;
  draftStrategy: DraftStrategy;
  verificationSensitivity: VerificationSensitivity;
  claims: ClaimViewModel[];
  highlightedSpans: HighlightSpan[];
  activeRewrite: RewriteAnimationState;
  verificationState: VerificationState;
  verificationError?: string | null;
};

export type DraftResponse = {
  message_id: string;
  query: string;
  draft_answer: string;
  answer_depth: AnswerDepth;
  insufficient_context: boolean;
  citations: Citation[];
  draft_model: string;
  verification_model: string;
  draft_strategy: DraftStrategy;
  verification_sensitivity: VerificationSensitivity;
};

export type VerificationEvent = {
  run_id: string;
  event_type: string;
  message_id: string | null;
  document_id: string | null;
  timestamp: string;
  payload: Record<string, unknown>;
};

export type DocumentRunResponse = {
  run_id: string;
  document_id: string;
};

export type DocumentEvent = {
  run_id: string;
  event_type: string;
  message_id: string | null;
  document_id: string | null;
  timestamp: string;
  payload: Record<string, unknown>;
};
