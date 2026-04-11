# Question
<question>
{query}
</question>

# Answer depth
<answer_depth>
{answer_depth}
</answer_depth>

# Draft strategy
<draft_strategy>
{draft_strategy}
</draft_strategy>

# Strategy note
<strategy_note>
{strategy_note}
</strategy_note>

# Content plan
<content_plan>
{content_plan}
</content_plan>

# Evidence
<evidence>
{evidence}
</evidence>

# Output schema
<output_schema>
{output_schema}
</output_schema>

# Task
Return a grounded answer using only the evidence above.
Follow the requested answer depth:
- `brief`: 1 short paragraph
- `standard`: 2-4 grounded sentences
- `detailed`: 3 or more short paragraphs and at least 6 grounded sentences if the evidence supports a richer explanation

For `detailed`, cover the topic by meaningful parts such as definition/framing, composition/stages/elements, and important clarifications that are explicitly supported by the evidence.
Do not repeat the same point in different words.
Follow the strategy note if one is provided.
Follow the content plan when it is provided.
If the evidence does not support a reliable answer, return `insufficient_context=true`.
Do not mention any source that is absent from the evidence list.
