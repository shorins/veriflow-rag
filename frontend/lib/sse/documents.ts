import { createRunEventSource } from "@/lib/api/client";
import type { DocumentEvent } from "@/lib/types";

export function subscribeToDocumentRun(
  runId: string,
  handlers: {
    onEvent: (event: DocumentEvent) => void;
    onError: (error: Event) => void;
    onDone: () => void;
  },
): () => void {
  const source = createRunEventSource(runId);
  const eventTypes = [
    "corpus_reindex_started",
    "corpus_reindex_progress",
    "corpus_reindex_completed",
    "corpus_error",
  ] as const;

  for (const eventType of eventTypes) {
    source.addEventListener(eventType, (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as DocumentEvent;
      handlers.onEvent(payload);
      if (eventType === "corpus_reindex_completed" || eventType === "corpus_error") {
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
