export function findSpanRange(text: string, span: string): [number, number] | null {
  if (!span) {
    return null;
  }

  const directIndex = text.indexOf(span);
  if (directIndex !== -1) {
    return [directIndex, directIndex + span.length];
  }

  const collapsedText = text.replace(/\s+/g, " ");
  const collapsedSpan = span.replace(/\s+/g, " ").trim();
  const collapsedIndex = collapsedText.indexOf(collapsedSpan);
  if (collapsedIndex === -1) {
    return null;
  }

  const strippedText = text.replace(/\s+/g, "");
  const strippedSpan = span.replace(/\s+/g, "");
  const strippedIndex = strippedText.indexOf(strippedSpan);
  if (strippedIndex === -1) {
    return null;
  }

  let seen = 0;
  let realStart = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (!/\s/.test(text[i])) {
      if (seen === strippedIndex) {
        realStart = i;
        break;
      }
      seen += 1;
    }
  }

  let realEnd = realStart;
  let matched = 0;
  for (let i = realStart; i < text.length; i += 1) {
    if (!/\s/.test(text[i])) {
      matched += 1;
    }
    if (matched === strippedSpan.length) {
      realEnd = i + 1;
      break;
    }
  }

  return [realStart, realEnd];
}
