# Question
<question>
{query}
</question>

# Evidence
<evidence>
{evidence}
</evidence>

# Output schema
<output_schema>
{output_schema}
</output_schema>

# Task
Return a concise grounded answer using only the evidence above.
If the evidence does not support a reliable answer, return `insufficient_context=true`.
Do not mention any source that is absent from the evidence list.
