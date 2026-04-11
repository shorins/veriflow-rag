import type { DocumentItem, DraftResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

type DraftRequest = {
  query: string;
  draft_model: string;
  verification_model: string;
  draft_strategy: "conservative" | "balanced" | "demo";
  verification_sensitivity: "conservative" | "balanced" | "demo";
};

type StartVerificationRequest = {
  query: string;
  draft_answer: string;
  answer_depth: "brief" | "standard" | "detailed";
  draft_model: string;
  verification_model: string;
  draft_strategy: "conservative" | "balanced" | "demo";
  verification_sensitivity: "conservative" | "balanced" | "demo";
};

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
  const response = await fetch(`${API_BASE_URL}/api/chat/draft`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJson<DraftResponse>(response);
}

export async function startVerification(
  messageId: string,
  payload: StartVerificationRequest,
): Promise<{ run_id: string; message_id: string }> {
  const response = await fetch(`${API_BASE_URL}/api/chat/${messageId}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJson<{ run_id: string; message_id: string }>(response);
}

export function createRunEventSource(runId: string): EventSource {
  return new EventSource(`${API_BASE_URL}/api/runs/${runId}/events`);
}

export { API_BASE_URL };
