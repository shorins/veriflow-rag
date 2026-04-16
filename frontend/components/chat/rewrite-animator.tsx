"use client";

import { useEffect, useMemo, useState, useRef } from "react";

import { findSpanRange } from "@/lib/text/find-span-range";
import type { HighlightSpan, RewriteAnimationState } from "@/lib/types";

type RenderHighlightSpan = HighlightSpan & {
  temporary?: boolean;
};

type Props = {
  text: string;
  highlights: HighlightSpan[];
  animation: RewriteAnimationState;
  activeClaimId?: string | null;
  activeClaimSpan?: string | null;
  onComplete?: (claimId: string) => void;
};

export function RewriteAnimator({ text, highlights, animation, activeClaimId, activeClaimSpan, onComplete }: Props) {
  const [displayText, setDisplayText] = useState(text);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    if (!animation) {
      setDisplayText(text);
      return;
    }

    const before = text;
    const oldSpan = animation.oldSpan;
    const newSpan = animation.newSpan;
    const spanRange = findSpanRange(before, oldSpan);
    if (!spanRange) {
      setDisplayText(text);
      if (onCompleteRef.current) {
        onCompleteRef.current(animation.claimId);
      }
      return;
    }
    const [startIndex, endIndex] = spanRange;

    const prefix = before.slice(0, startIndex);
    const matchedOldSpan = before.slice(startIndex, endIndex);
    const suffix = before.slice(endIndex);
    let eraseIndex = matchedOldSpan.length;
    let typeIndex = 0;
    let typeTimer: number | null = null;

    if (animation.phase === "done") {
      setDisplayText(prefix + newSpan + suffix);
      return;
    }

    const eraseTimer = window.setInterval(() => {
      eraseIndex -= 1;
      setDisplayText(prefix + matchedOldSpan.slice(0, Math.max(eraseIndex, 0)) + suffix);
      if (eraseIndex <= 0) {
        window.clearInterval(eraseTimer);
        if (animation.phase === "erasing") {
          return;
        }
        typeTimer = window.setInterval(() => {
          typeIndex += 1;
          setDisplayText(prefix + newSpan.slice(0, typeIndex) + suffix);
          if (typeIndex >= newSpan.length) {
            if (typeTimer !== null) {
              window.clearInterval(typeTimer);
            }
            if (onCompleteRef.current) {
              onCompleteRef.current(animation.claimId);
            }
          }
        }, 12);
      }
    }, 12);

    return () => {
      window.clearInterval(eraseTimer);
      if (typeTimer !== null) {
        window.clearInterval(typeTimer);
      }
    };
  }, [animation, text]);

  return (
    <HighlightedText
      text={displayText}
      highlights={highlights}
      activeClaimId={activeClaimId ?? null}
      activeClaimSpan={activeClaimSpan ?? null}
    />
  );
}

function HighlightedText({
  text,
  highlights,
  activeClaimId,
  activeClaimSpan,
}: {
  text: string;
  highlights: HighlightSpan[];
  activeClaimId: string | null;
  activeClaimSpan: string | null;
}) {
  const temporaryActiveHighlight = useMemo(() => {
    if (!activeClaimId || !activeClaimSpan) {
      return null;
    }
    const alreadyTracked = highlights.some((item) => item.claimId === activeClaimId);
    if (alreadyTracked) {
      return null;
    }
    const temporaryHighlight: RenderHighlightSpan = {
      claimId: activeClaimId,
      sourceSpan: activeClaimSpan,
      status: "partial" as const,
      reason: "",
      isActive: true,
      temporary: true,
    };
    return temporaryHighlight;
  }, [activeClaimId, activeClaimSpan, highlights]);

  const sorted = useMemo(
    () =>
      ([...highlights, ...(temporaryActiveHighlight ? [temporaryActiveHighlight] : [])] as RenderHighlightSpan[])
        .map((item) => {
          const spanRange = findSpanRange(text, item.sourceSpan);
          return { ...item, index: spanRange?.[0] ?? -1, end: spanRange?.[1] ?? -1 };
        })
        .filter((item) => item.index >= 0)
        .sort((a, b) => a.index - b.index),
    [highlights, temporaryActiveHighlight, text],
  );

  if (!sorted.length) {
    return <p className="whitespace-pre-wrap leading-7">{text}</p>;
  }

  const parts: React.ReactNode[] = [];
  let cursor = 0;
  for (const item of sorted) {
    if (item.index > cursor) {
      parts.push(<span key={`${item.claimId}-before`}>{text.slice(cursor, item.index)}</span>);
    }
    const value = text.slice(item.index, item.end);
    const badgeClass = item.temporary
      ? "rounded-md bg-stone-300/55 px-1 py-0.5 animate-claim-pulse"
      : item.status === "partial"
        ? "rounded-md bg-amber-200/80 px-1 py-0.5"
        : item.status === "unsupported"
          ? "rounded-md bg-rose-200/85 px-1 py-0.5"
          : "rounded-md bg-rose-400/25 px-1 py-0.5 line-through decoration-rose-700";
    const activeClass = item.isActive
      ? " ring-1 ring-stone-400/55 shadow-[0_0_0_1px_rgba(120,113,108,0.08)] animate-claim-pulse"
      : "";
    const tooltipTitle = item.temporary ? "Проверяется сейчас" : statusLabel(item.status);
    const tooltipBody = item.temporary ? "Система сейчас проверяет этот claim по найденным evidence." : item.reason;

    parts.push(
      <span key={item.claimId} className="group relative inline">
        <span className={`${badgeClass}${activeClass}`}>{value}</span>
        <span className="pointer-events-none absolute bottom-[calc(100%+10px)] left-1/2 z-20 hidden w-72 -translate-x-1/2 rounded-2xl border border-stone-200 bg-white px-3 py-2 text-left shadow-xl group-hover:block">
          <span className="block text-[11px] font-semibold uppercase tracking-[0.18em] text-stone-500">
            {tooltipTitle}
          </span>
          {tooltipBody ? <span className="mt-1 block text-xs leading-5 text-stone-700">{tooltipBody}</span> : null}
        </span>
      </span>,
    );
    cursor = item.end;
  }
  if (cursor < text.length) {
    parts.push(<span key="tail">{text.slice(cursor)}</span>);
  }
  return <p className="whitespace-pre-wrap leading-7">{parts}</p>;
}

function statusLabel(status: HighlightSpan["status"]) {
  if (status === "partial") {
    return "Частично подтверждено";
  }
  if (status === "unsupported") {
    return "Не подтверждено";
  }
  return "Противоречит evidence";
}
