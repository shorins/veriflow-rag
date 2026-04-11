"use client";

import { useMemo } from "react";
import { useDropzone } from "react-dropzone";
import { FileUp, RefreshCw, Trash2 } from "lucide-react";

import type { DocumentItem } from "@/lib/types";

type Props = {
  documents: DocumentItem[];
  onUpload: (files: File[]) => void;
  onDelete: (documentId: string) => void;
  onReindexCorpus: () => void;
  corpusRunState: "idle" | "running" | "error" | "completed";
  corpusRunLabel?: string | null;
};

const statusTone: Record<DocumentItem["status"], string> = {
  uploaded: "bg-stone-100 text-stone-700",
  indexed: "bg-emerald-100 text-emerald-700",
  stale: "bg-orange-100 text-orange-700",
  error: "bg-rose-100 text-rose-700",
};

export function DocumentSidebar({ documents, onUpload, onDelete, onReindexCorpus, corpusRunState, corpusRunLabel }: Props) {
  const dropzone = useDropzone({
    accept: { "application/pdf": [".pdf"] },
    multiple: true,
    onDrop: (files) => {
      if (files.length) {
        onUpload(files);
      }
    },
  });

  const totalDocs = useMemo(() => documents.length, [documents.length]);
  const needsReindex = useMemo(
    () => documents.some((doc) => doc.status === "uploaded" || doc.status === "stale" || doc.status === "error"),
    [documents],
  );
  const indexedCount = useMemo(() => documents.filter((doc) => doc.status === "indexed").length, [documents]);

  return (
    <aside className="flex h-full min-h-0 flex-col gap-4 rounded-3xl border border-stone-200 bg-panel p-4 shadow-panel">
      <div>
        <p className="text-xs uppercase tracking-[0.24em] text-mutedink">Corpus</p>
        <h2 className="mt-2 text-xl font-semibold">Документы</h2>
        <p className="mt-1 text-sm text-mutedink">
          Загружайте PDF в корпус, затем пересобирайте retrieval-индекс сразу по всем документам из <span className="font-mono">data/</span>.
        </p>
      </div>

      <button
        {...dropzone.getRootProps()}
        className="rounded-2xl border border-dashed border-stone-300 bg-canvas px-4 py-5 text-left transition hover:border-accent hover:bg-teal-50/50"
      >
        <input {...dropzone.getInputProps()} />
        <div className="flex items-start gap-3">
          <FileUp className="mt-0.5 size-5 text-accent" />
          <div>
            <p className="font-medium">Upload PDF</p>
            <p className="mt-1 text-sm text-mutedink">Drag-and-drop или нажмите, чтобы выбрать файлы.</p>
          </div>
        </div>
      </button>

      <div className="flex items-center justify-between text-sm text-mutedink">
        <span>{totalDocs} file(s)</span>
        <span className="font-mono">data/</span>
      </div>

      <div className="rounded-2xl border border-stone-200 bg-canvas p-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-sm font-medium">Индекс корпуса</p>
            <p className="mt-1 text-xs text-mutedink">
              {needsReindex
                ? "Корпус изменился. После загрузки или удаления файлов нужно обновить индекс."
                : "Индекс актуален для текущего набора PDF."}
            </p>
          </div>
          <span
            className={`rounded-full px-2 py-1 text-[11px] font-medium ${
              needsReindex ? "bg-amber-100 text-amber-700" : "bg-emerald-100 text-emerald-700"
            }`}
          >
            {needsReindex ? "needs update" : "up to date"}
          </span>
        </div>

        <div className="mt-3 flex items-center justify-between gap-3 text-xs text-mutedink">
          <span>{indexedCount}/{totalDocs} indexed</span>
          <span>Индекс строится по всем PDF в data/</span>
        </div>

        <button
          className={`mt-4 inline-flex w-full items-center justify-center gap-2 rounded-2xl px-4 py-2.5 text-sm font-medium transition ${
            needsReindex
              ? "bg-ink text-white hover:bg-zinc-800"
              : "border border-stone-200 bg-white text-stone-800 hover:bg-stone-50"
          } disabled:cursor-not-allowed disabled:opacity-50`}
          disabled={corpusRunState === "running" || !documents.length}
          onClick={onReindexCorpus}
        >
          <RefreshCw className={`size-4 ${corpusRunState === "running" ? "animate-spin" : ""}`} />
          Обновить индекс
        </button>

        {corpusRunLabel ? <p className="mt-3 text-xs text-accent">{corpusRunLabel}</p> : null}
      </div>

      <div className="scrollbar-subtle flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
        {documents.map((doc) => {
          return (
            <div key={doc.document_id} className="rounded-2xl border border-stone-200 bg-canvas p-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{doc.file_name}</p>
                  <p className="mt-1 text-xs text-mutedink">{formatBytes(doc.size_bytes)}</p>
                </div>
                <span className={`rounded-full px-2 py-1 text-[11px] font-medium ${statusTone[doc.status]}`}>
                  {doc.status}
                </span>
              </div>

              {doc.error_message ? (
                <p className="mt-2 text-xs text-rose-600">{doc.error_message}</p>
              ) : null}

              <div className="mt-3 grid grid-cols-1 gap-2">
                <ActionButton tone="danger" disabled={corpusRunState === "running"} onClick={() => onDelete(doc.document_id)}>
                  <Trash2 className="size-3.5" />
                  Delete
                </ActionButton>
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}

function ActionButton(props: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  tone?: "default" | "danger";
}) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-1 rounded-xl px-3 py-2 text-sm transition ${
        props.tone === "danger"
          ? "border border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100"
          : "border border-stone-200 bg-white text-stone-800 hover:bg-stone-50"
      } disabled:cursor-not-allowed disabled:opacity-50`}
      disabled={props.disabled}
      onClick={props.onClick}
    >
      {props.children}
    </button>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
