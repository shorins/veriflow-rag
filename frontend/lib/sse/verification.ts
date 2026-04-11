import { createRunEventSource } from "@/lib/api/client";
import type { VerificationEvent } from "@/lib/types";

export function subscribeToVerificationRun(
  runId: string,
  handlers: {
    onEvent: (event: VerificationEvent) => void;
    onError: (error: Event) => void;
    onDone: () => void;
  },
): () => void {
  const source = createRunEventSource(runId);
  const eventTypes = [
    "verification_started",
    "claims_extracted",
    "claim_started",
    "claim_supported",
    "claim_partial",
    "claim_unsupported",
    "claim_contradicted",
    "rewrite_started",
    "rewrite_span_erasing",
    "rewrite_span_typing",
    "rewrite_finished",
    "verification_completed",
    "run_error",
  ] as const;

  for (const eventType of eventTypes) {
    source.addEventListener(eventType, (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as VerificationEvent;
      handlers.onEvent(payload);
      if (eventType === "verification_completed" || eventType === "run_error") {
        handlers.onDone();
        source.close();
      }
    });
  }

  source.onerror = (error) => {
    handlers.onError(error);
    source.close();
  };

  return () => source.close();
}

