# Query
<query>
{query}
</query>

# Grounded draft answer
<draft_answer>
{draft_answer}
</draft_answer>

# Draft sentences
<sentence_list>
{sentence_list}
</sentence_list>

# Evidence
<evidence>
{evidence}
</evidence>

# Task
Create up to {fault_count} controlled mismatches for demo visualization.

Choose sentences from the grounded draft answer and rewrite them into slightly wrong but still plausible variants that are not fully supported by the evidence.

Priorities:
- prefer topic-consistent local errors
- prefer errors that a verifier can catch from the provided evidence
- do not introduce unrelated concepts from a different domain
- if the evidence is too weak for a good perturbation, return fewer faults

For each fault:
- `original_span` must match the original sentence exactly
- `injected_span` must be the rewritten sentence that will replace it
- `source_sentence_index` must match the sentence list
- keep the rewritten sentence fluent and natural

Return valid JSON only and follow this schema:
```json
{output_schema}
```
