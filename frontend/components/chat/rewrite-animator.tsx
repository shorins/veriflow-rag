"use client";

import { useEffect, useMemo, useState } from "react";

import type { HighlightSpan, RewriteAnimationState } from "@/lib/types";

type Props = {
  text: string;
  highlights: HighlightSpan[];
  animation: RewriteAnimationState;
};

export function RewriteAnimator({ text, highlights, animation }: Props) {
  const [displayText, setDisplayText] = useState(text);

  useEffect(() => {
    if (!animation) {
      setDisplayText(text);
      return;
    }

    const before = text;
    const oldSpan = animation.oldSpan;
    const newSpan = animation.newSpan;
    const startIndex = before.indexOf(oldSpan);
    if (startIndex === -1) {
      setDisplayText(text);
      return;
    }

    const prefix = before.slice(0, startIndex);
    const suffix = before.slice(startIndex + oldSpan.length);
    let eraseIndex = oldSpan.length;
    let typeIndex = 0;
    let typeTimer: number | null = null;

    if (animation.phase === "done") {
      setDisplayText(before.replace(oldSpan, newSpan));
      return;
    }

    const eraseTimer = window.setInterval(() => {
      eraseIndex -= 1;
      setDisplayText(prefix + oldSpan.slice(0, Math.max(eraseIndex, 0)) + suffix);
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

  return <HighlightedText text={displayText} highlights={highlights} />;
}

function HighlightedText({ text, highlights }: { text: string; highlights: HighlightSpan[] }) {
  const sorted = useMemo(
    () =>
      [...highlights]
    .map((item) => {
      const index = text.indexOf(item.sourceSpan);
      return { ...item, index };
    })
    .filter((item) => item.index >= 0)
    .sort((a, b) => a.index - b.index),
    [highlights, text],
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
    const value = text.slice(item.index, item.index + item.sourceSpan.length);
    parts.push(
      <span
        key={item.claimId}
        className={
          item.status === "partial"
            ? "rounded-md bg-amber-200/80 px-1 py-0.5"
            : item.status === "unsupported"
              ? "rounded-md bg-rose-200/85 px-1 py-0.5"
              : "rounded-md bg-rose-400/25 px-1 py-0.5 line-through decoration-rose-700"
        }
      >
        {value}
      </span>,
    );
    cursor = item.index + item.sourceSpan.length;
  }
  if (cursor < text.length) {
    parts.push(<span key="tail">{text.slice(cursor)}</span>);
  }
  return <p className="whitespace-pre-wrap leading-7">{parts}</p>;
}
