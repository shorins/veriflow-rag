import type { DemoFaultMode, DocumentItem, DraftResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const UI_TEST_HEADER = "X-TrustRAG-UI-Test";

type DraftRequest = {
  query: string;
  draft_model: string;
  verification_model: string;
  draft_strategy: "conservative" | "balanced" | "demo";
  verification_sensitivity: "conservative" | "balanced" | "demo";
  demo_fault_mode: DemoFaultMode;
  demo_fault_count: number;
  ui_test_mode?: boolean;
};

type StartVerificationRequest = {
  query: string;
  draft_answer: string;
  answer_depth: "brief" | "standard" | "detailed";
  draft_model: string;
  verification_model: string;
  draft_strategy: "conservative" | "balanced" | "demo";
  verification_sensitivity: "conservative" | "balanced" | "demo";
  demo_fault_mode: DemoFaultMode;
  demo_fault_count: number;
  ui_test_mode?: boolean;
};

function withUiTest(url: string, enabled?: boolean): string {
  if (!enabled) {
    return url;
  }
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}ui_test=1`;
}

function buildHeaders(uiTestMode?: boolean): HeadersInit {
  return uiTestMode ? { "Content-Type": "application/json", [UI_TEST_HEADER]: "1" } : { "Content-Type": "application/json" };
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchDocuments(): Promise<DocumentItem[]> {
  const response = await fetch(`${API_BASE_URL}/api/documents`, { cache: "no-store" });
  const data = await readJson<{ items: DocumentItem[] }>(response);
  return data.items;
}

export async function uploadDocuments(files: File[]): Promise<DocumentItem[]> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
    method: "POST",
    body: formData,
  });
  const data = await readJson<{ items: DocumentItem[] }>(response);
  return data.items;
}

export async function startCorpusReindex(): Promise<{ run_id: string }> {
  const response = await fetch(`${API_BASE_URL}/api/corpus/reindex/run`, {
    method: "POST",
  });
  return readJson<{ run_id: string }>(response);
}

export async function deleteDocument(documentId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Delete failed with status ${response.status}`);
  }
}

export async function createDraft(payload: DraftRequest): Promise<DraftResponse> {
  const response = await fetch(withUiTest(`${API_BASE_URL}/api/chat/draft`, payload.ui_test_mode), {
    method: "POST",
    headers: buildHeaders(payload.ui_test_mode),
    body: JSON.stringify(payload),
  });
  return readJson<DraftResponse>(response);
}

export async function startVerification(
  messageId: string,
  payload: StartVerificationRequest,
): Promise<{ run_id: string; message_id: string }> {
  const response = await fetch(withUiTest(`${API_BASE_URL}/api/chat/${messageId}/verify`, payload.ui_test_mode), {
    method: "POST",
    headers: buildHeaders(payload.ui_test_mode),
    body: JSON.stringify(payload),
  });
  return readJson<{ run_id: string; message_id: string }>(response);
}

export function createRunEventSource(runId: string, uiTestMode?: boolean): EventSource {
  return new EventSource(withUiTest(`${API_BASE_URL}/api/runs/${runId}/events`, uiTestMode));
}

export { API_BASE_URL };
