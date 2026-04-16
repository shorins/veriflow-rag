"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { ChatPanel } from "@/components/chat/chat-panel";
import { DocumentSidebar } from "@/components/documents/document-sidebar";
import { ControlBar } from "@/components/layout/control-bar";
import { deleteDocument, fetchDocuments, startCorpusReindex, uploadDocuments } from "@/lib/api/client";
import { subscribeToDocumentRun } from "@/lib/sse/documents";
import type { DemoFaultMode, DraftStrategy, VerificationSensitivity } from "@/lib/types";

export default function HomePage() {
  const queryClient = useQueryClient();
  const [uiTestMode, setUiTestMode] = useState(false);
  const [draftModel, setDraftModel] = useState("qwen2.5-vl-3b-instruct");
  const [verificationModel, setVerificationModel] = useState("qwen2.5-vl-7b-instruct");
  const [draftStrategy, setDraftStrategy] = useState<DraftStrategy>("demo");
  const [verificationSensitivity, setVerificationSensitivity] = useState<VerificationSensitivity>("demo");
  const [demoFaultMode, setDemoFaultMode] = useState<DemoFaultMode>("off");
  const [demoFaultCount, setDemoFaultCount] = useState(1);
  const [clearChatSignal, setClearChatSignal] = useState(0);
  const [corpusRunState, setCorpusRunState] = useState<"idle" | "running" | "error" | "completed">("idle");
  const [corpusRunLabel, setCorpusRunLabel] = useState<string | null>(null);

  useEffect(() => {
    setUiTestMode(new URLSearchParams(window.location.search).get("uiTest") === "1");
  }, []);

  const documentsQuery = useQuery({
    queryKey: ["documents"],
    queryFn: fetchDocuments,
  });

  const uploadMutation = useMutation({
    mutationFn: uploadDocuments,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  async function handleDeleteDocument(documentId: string) {
    try {
      setCorpusRunLabel("Удаление файла из корпуса...");
      await deleteDocument(documentId);
      await queryClient.invalidateQueries({ queryKey: ["documents"] });
    } finally {
      setCorpusRunLabel(null);
    }
  }

  async function handleReindexCorpus() {
    setCorpusRunState("running");
    setCorpusRunLabel("Запускаю обновление индекса корпуса...");
    let failed = false;
    try {
      const start = await startCorpusReindex();
      await new Promise<void>((resolve) => {
        subscribeToDocumentRun(start.run_id, {
          onEvent: (event) => {
            if (event.event_type === "corpus_reindex_started") {
              setCorpusRunLabel("Начинаю пересборку индекса по всем PDF...");
            }
            if (event.event_type === "corpus_reindex_progress") {
              const phase = event.payload.phase as string | undefined;
              if (phase === "sync_registry") {
                setCorpusRunLabel("Синхронизирую состав корпуса...");
              } else if (phase === "rebuild_manifest") {
                setCorpusRunLabel("Пересобираю manifest и чанки...");
              } else if (phase === "reindex_weaviate") {
                setCorpusRunLabel("Обновляю индекс в Weaviate...");
              } else if (phase === "finalize_registry") {
                setCorpusRunLabel("Финализирую статусы документов...");
              }
            }
            if (event.event_type === "corpus_reindex_completed") {
              setCorpusRunState("completed");
              setCorpusRunLabel("Индекс корпуса обновлён.");
              void queryClient.invalidateQueries({ queryKey: ["documents"] });
            }
            if (event.event_type === "corpus_error") {
              failed = true;
              setCorpusRunState("error");
              setCorpusRunLabel(String(event.payload.message ?? "Ошибка при обновлении индекса."));
              void queryClient.invalidateQueries({ queryKey: ["documents"] });
            }
          },
          onError: () => {
            failed = true;
            setCorpusRunState("error");
            setCorpusRunLabel("Ошибка SSE при обновлении индекса.");
            resolve();
          },
          onDone: () => {
            void queryClient.invalidateQueries({ queryKey: ["documents"] });
            resolve();
          },
        });
      });
    } finally {
      if (!failed) {
        setCorpusRunState("idle");
      }
    }
  }

  return (
    <main className="min-h-screen bg-canvas px-5 py-5 text-ink lg:px-6 lg:py-6">
      <div className="grid min-h-[calc(100vh-2.5rem)] gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
        <DocumentSidebar
          documents={documentsQuery.data ?? []}
          onUpload={(files) => uploadMutation.mutate(files)}
          onDelete={handleDeleteDocument}
          onReindexCorpus={handleReindexCorpus}
          corpusRunState={corpusRunState}
          corpusRunLabel={corpusRunLabel}
        />

        <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-4">
          <ControlBar
            draftModel={draftModel}
            verificationModel={verificationModel}
            draftStrategy={draftStrategy}
            verificationSensitivity={verificationSensitivity}
            demoFaultMode={demoFaultMode}
            demoFaultCount={demoFaultCount}
            uiTestMode={uiTestMode}
            onDraftModelChange={setDraftModel}
            onVerificationModelChange={setVerificationModel}
            onDraftStrategyChange={setDraftStrategy}
            onVerificationSensitivityChange={setVerificationSensitivity}
            onDemoFaultModeChange={setDemoFaultMode}
            onDemoFaultCountChange={setDemoFaultCount}
            onClearChat={() => setClearChatSignal((value) => value + 1)}
          />
          <ChatPanel
            draftModel={draftModel}
            verificationModel={verificationModel}
            draftStrategy={draftStrategy}
            verificationSensitivity={verificationSensitivity}
            demoFaultMode={demoFaultMode}
            demoFaultCount={demoFaultCount}
            uiTestMode={uiTestMode}
            clearChatSignal={clearChatSignal}
          />
        </div>
      </div>
    </main>
  );
}
