"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";

import { createDraft, startVerification } from "@/lib/api/client";
import { subscribeToVerificationRun } from "@/lib/sse/verification";
import { applyVerificationEvent, createDraftMessage, completeActiveRewrite } from "@/lib/state/message-state";
import type { DemoFaultMode, DraftMessage, DraftStrategy, VerificationSensitivity } from "@/lib/types";
import { RewriteAnimator } from "@/components/chat/rewrite-animator";

type Props = {
  draftModel: string;
  verificationModel: string;
  draftStrategy: DraftStrategy;
  verificationSensitivity: VerificationSensitivity;
  demoFaultMode: DemoFaultMode;
  demoFaultCount: number;
  uiTestMode: boolean;
  clearChatSignal: number;
};

export function ChatPanel(props: Props) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<DraftMessage[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const storageKey = props.uiTestMode ? "trustrag-chat-history-v2-ui-test" : "trustrag-chat-history-v2";

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (!raw) {
        setMessages([]);
        return;
      }
      const parsed = JSON.parse(raw) as DraftMessage[];
      setMessages(parsed);
    } catch {
      setMessages([]);
    }
  }, [storageKey]);

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(messages));
    } catch {
      // Ignore persistence failures in private/incognito contexts.
    }
  }, [messages]);

  useEffect(() => {
    if (props.clearChatSignal === 0) {
      return;
    }
    setMessages([]);
    try {
      window.localStorage.removeItem(storageKey);
    } catch {
      // Ignore storage failures and still clear in-memory state.
    }
  }, [props.clearChatSignal]);

  const canSubmit = query.trim().length > 0 && !isSubmitting;

  async function handleSubmit() {
    const trimmed = query.trim();
    if (!trimmed) return;
    setIsSubmitting(true);
    try {
      const response = await createDraft({
        query: trimmed,
        draft_model: props.draftModel,
        verification_model: props.verificationModel,
        draft_strategy: props.draftStrategy,
        verification_sensitivity: props.verificationSensitivity,
        demo_fault_mode: props.demoFaultMode,
        demo_fault_count: props.demoFaultCount,
        ui_test_mode: props.uiTestMode,
      });
      const message = createDraftMessage({
        id: response.message_id,
        query: response.query,
        draftText: response.draft_answer,
        answerDepth: response.answer_depth,
        insufficientContext: response.insufficient_context,
        citations: response.citations,
        draftModel: response.draft_model,
        verificationModel: response.verification_model,
        draftStrategy: response.draft_strategy,
        verificationSensitivity: response.verification_sensitivity,
        groundedAnswer: response.grounded_answer,
        faultInjectionActive: response.fault_injection_active,
        demoFaultMode: response.demo_fault_mode,
        demoFaultCount: response.demo_fault_count,
        faultInjectionSummary: response.fault_injection_summary,
      });
      setMessages((current) => [...current, message]);
      setQuery("");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleVerify(message: DraftMessage) {
    const start = await startVerification(message.id, {
      query: message.query,
      draft_answer: message.draftText,
      answer_depth: message.answerDepth,
      draft_model: message.draftModel,
      verification_model: message.verificationModel,
      draft_strategy: message.draftStrategy,
      verification_sensitivity: message.verificationSensitivity,
      demo_fault_mode: message.demoFaultMode,
      demo_fault_count: message.demoFaultCount,
      ui_test_mode: props.uiTestMode,
    });

    setMessages((current) =>
      current.map((item) => (item.id === message.id ? { ...item, verificationState: "running" } : item)),
    );

    subscribeToVerificationRun(start.run_id, props.uiTestMode, {
      onEvent: (event) => {
        setMessages((current) =>
          current.map((item) => (item.id === message.id ? applyVerificationEvent(item, event) : item)),
        );
      },
      onError: () => {
        setMessages((current) =>
          current.map((item) =>
            item.id === message.id ? { ...item, verificationState: "error", verificationError: "SSE connection failed." } : item,
          ),
        );
      },
      onDone: () => {},
    });
  }

  const emptyState = useMemo(
    () =>
      !messages.length ? (
        <div className="rounded-3xl border border-dashed border-stone-300 bg-panel px-8 py-10 text-center shadow-panel">
          <p className="text-sm uppercase tracking-[0.24em] text-mutedink">trustRAG</p>
          <h2 className="mt-3 text-2xl font-semibold">Live grounded chat and verification</h2>
          <p className="mt-3 text-sm text-mutedink">
            Сначала получаем draft answer, затем локально запускаем агентную проверку claims и показываем rewrite in-place.
          </p>
        </div>
      ) : null,
    [messages.length],
  );

  return (
    <section className="flex h-full min-h-0 flex-col gap-4">
      <div className="scrollbar-subtle flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pr-1">
        {emptyState}
        {messages.map((message) => (
          <motion.article
            key={message.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-3xl border border-stone-200 bg-panel p-5 shadow-panel"
          >
            <p className="text-xs uppercase tracking-[0.24em] text-mutedink">User query</p>
            <p className="mt-2 text-sm text-stone-700">{message.query}</p>

            <div className="mt-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-mutedink">Draft answer</p>
                  <p className="mt-1 text-sm text-mutedink">
                    {message.draftModel} · {message.answerDepth} · {message.draftStrategy}
                  </p>
                  {message.faultInjectionActive ? (
                    <p className="mt-1 text-xs text-amber-700">{message.faultInjectionSummary}</p>
                  ) : null}
                </div>
                {!message.insufficientContext ? (
                  <button
                    className="rounded-xl border border-stone-200 bg-white px-4 py-2 text-sm font-medium transition hover:bg-stone-50 disabled:opacity-50"
                    disabled={message.verificationState === "running" || message.verificationState === "rewriting"}
                    onClick={() => handleVerify(message)}
                  >
                    Агентная проверка
                  </button>
                ) : null}
              </div>
              <div className="mt-4 rounded-2xl bg-canvas p-4">
                <RewriteAnimator
                  text={message.displayText}
                  highlights={message.highlightedSpans}
                  animation={message.activeRewrite}
                  activeClaimId={message.activeClaimId}
                  activeClaimSpan={message.activeClaimSpan}
                  onComplete={() => {
                    setMessages((current) =>
                      current.map((item) => (item.id === message.id ? completeActiveRewrite(item) : item)),
                    );
                  }}
                />
              </div>
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.9fr)]">
              <div className="rounded-2xl border border-stone-200 bg-canvas p-4">
                <p className="text-xs uppercase tracking-[0.24em] text-mutedink">Citations</p>
                <div className="mt-3 space-y-3 text-sm">
                  {message.citations.map((citation, index) => (
                    <div
                      key={`${citation.evidence_id}-${citation.file_name}-${index}`}
                      className="rounded-xl border border-stone-200 bg-white p-3"
                    >
                      <p className="font-medium">
                        [{citation.evidence_id}] {citation.file_name}
                      </p>
                      <p className="mt-1 text-xs text-mutedink">{citation.section_title}</p>
                      <p className="mt-2 text-sm text-stone-700">{citation.support}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border border-stone-200 bg-canvas p-4">
                <p className="text-xs uppercase tracking-[0.24em] text-mutedink">Verification</p>
                <p className="mt-2 text-sm text-mutedink">
                  {message.verificationState === "idle"
                    ? "Проверка ещё не запускалась."
                    : message.verificationState === "running"
                      ? "Claims проверяются последовательно."
                      : message.verificationState === "rewriting"
                        ? "Идёт локальное переписывание проблемных фрагментов."
                        : message.verificationState === "completed"
                          ? "Проверка завершена."
                          : `Ошибка: ${message.verificationError ?? "unknown"}`}
                </p>

                <div className="mt-4 space-y-3">
                  {message.claims.map((claim) => (
                    <div key={claim.claim_id} className="rounded-xl border border-stone-200 bg-white p-3">
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-mono text-xs text-mutedink">{claim.claim_id}</span>
                        <span
                          className={`rounded-full px-2 py-1 text-[11px] font-medium ${
                            claim.status === "supported"
                              ? "bg-stone-100 text-stone-700"
                              : claim.status === "partial"
                                ? "bg-amber-100 text-amber-700"
                                : claim.status === "unsupported"
                                  ? "bg-rose-100 text-rose-700"
                                  : "bg-rose-200 text-rose-800"
                          }`}
                        >
                          {claim.status}
                        </span>
                      </div>
                      <p className="mt-2 text-sm">{claim.claim_text}</p>
                      <p className="mt-2 text-xs text-mutedink">{claim.reason}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.article>
        ))}
      </div>

      <div className="rounded-3xl border border-stone-200 bg-panel p-4 shadow-panel">
        <label className="text-xs uppercase tracking-[0.24em] text-mutedink">Prompt</label>
        <textarea
          className="mt-3 min-h-28 w-full resize-none rounded-2xl border border-stone-200 bg-canvas px-4 py-3 outline-none transition focus:border-stone-400"
          placeholder="Задайте вопрос по загруженному корпусу..."
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <div className="mt-3 flex items-center justify-between gap-3">
          <p className="text-sm text-mutedink">Draft: {props.draftModel} · Verification: {props.verificationModel}</p>
          <button
            className="rounded-2xl bg-ink px-5 py-2.5 text-sm font-medium text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!canSubmit}
            onClick={handleSubmit}
          >
            {isSubmitting ? "Генерация..." : "Отправить"}
          </button>
        </div>
      </div>
    </section>
  );
}
