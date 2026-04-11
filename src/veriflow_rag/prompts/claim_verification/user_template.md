# Claim
<claim>
id: {claim_id}
text: {claim_text}
source_span: {source_span}
</claim>

# Verification sensitivity
<verification_sensitivity>
{verification_sensitivity}
</verification_sensitivity>

# Sensitivity note
<sensitivity_note>
{sensitivity_note}
</sensitivity_note>

# Evidence
<evidence>
{evidence}
</evidence>

# Output schema
<output_schema>
{output_schema}
</output_schema>

# Task
Verify the claim using only the provided evidence and return one verification result.
If a sensitivity note is provided, apply it when choosing the status and rewrite decision.
Write `reason` in the same language as the claim text.
