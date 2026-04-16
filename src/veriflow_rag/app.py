from __future__ import annotations

import asyncio

import chainlit as cl
from chainlit.input_widget import Select

from veriflow_rag.core.config import AppConfig, get_config
from veriflow_rag.demo.fault_injection import apply_demo_verification_override, merge_injected_claims
from veriflow_rag.synthesis.client import LMStudioClientError
from veriflow_rag.synthesis.models import SynthesisResultBundle
from veriflow_rag.synthesis.service import build_synthesis_service
from veriflow_rag.ui.render import (
    render_claim_table,
    render_highlighted_answer,
    render_rewrite_diff,
)
from veriflow_rag.verification.orchestrator import build_verification_orchestrator
from veriflow_rag.verification.rewrite import (
    apply_rewrites,
    select_rewrite_span,
    select_verification_profile,
    should_trigger_rewrite,
)


SESSION_DRAFT_BUNDLE = "draft_bundle"
SESSION_VERIFICATION_RUNNING = "verification_running"
SESSION_DRAFT_MODEL = "draft_model"
SESSION_VERIFICATION_MODEL = "verification_model"
SESSION_DRAFT_STRATEGY = "draft_strategy"
SESSION_VERIFICATION_SENSITIVITY = "verification_sensitivity"
SESSION_DEMO_FAULT_MODE = "demo_fault_mode"
SESSION_DEMO_FAULT_COUNT = "demo_fault_count"
SETTINGS_DRAFT_MODEL = "draft_model"
SETTINGS_VERIFICATION_MODEL = "verification_model"
SETTINGS_DRAFT_STRATEGY = "draft_strategy"
SETTINGS_VERIFICATION_SENSITIVITY = "verification_sensitivity"
SETTINGS_DEMO_FAULT_MODE = "demo_fault_mode"
SETTINGS_DEMO_FAULT_COUNT = "demo_fault_count"


def _normalize_span(text: str | None) -> str:
    return " ".join((text or "").split()).strip().lower()


def _base_config() -> AppConfig:
    return get_config()


def _selected_draft_model_name() -> str:
    session_model = cl.user_session.get(SESSION_DRAFT_MODEL)
    config = _base_config()
    if session_model in config.available_draft_models:
        return session_model
    return config.draft_model_name


def _selected_verification_model_name() -> str:
    session_model = cl.user_session.get(SESSION_VERIFICATION_MODEL)
    config = _base_config()
    if session_model in config.available_verification_models:
        return session_model
    return config.verification_model_name


def _selected_draft_strategy() -> str:
    return str(cl.user_session.get(SESSION_DRAFT_STRATEGY) or _base_config().draft_strategy)


def _selected_verification_sensitivity() -> str:
    return str(cl.user_session.get(SESSION_VERIFICATION_SENSITIVITY) or _base_config().verification_sensitivity)


def _selected_demo_fault_mode() -> str:
    return str(cl.user_session.get(SESSION_DEMO_FAULT_MODE) or _base_config().demo_fault_mode)


def _selected_demo_fault_count() -> int:
    return int(cl.user_session.get(SESSION_DEMO_FAULT_COUNT) or _base_config().demo_fault_count)


def _selected_config() -> AppConfig:
    config = _base_config()
    return (
        config.with_draft_model(_selected_draft_model_name())
        .with_verification_model(_selected_verification_model_name())
        .with_draft_strategy(_selected_draft_strategy())
        .with_verification_sensitivity(_selected_verification_sensitivity())
        .with_demo_fault_mode(_selected_demo_fault_mode())
        .with_demo_fault_count(_selected_demo_fault_count())
    )


def _render_draft_markdown(bundle: SynthesisResultBundle) -> str:
    answer = bundle.synthesized_answer
    lines = [
        "## Draft Answer",
        "",
        answer.answer,
        "",
        f"- `draft_model`: `{answer.model_name}`",
        f"- `answer_depth`: `{answer.answer_depth}`",
        f"- `insufficient_context`: `{answer.insufficient_context}`",
        f"- `draft_strategy`: `{_selected_draft_strategy()}`",
        f"- `verification_model`: `{_selected_verification_model_name()}`",
        f"- `verification_sensitivity`: `{_selected_verification_sensitivity()}`",
        f"- `demo_fault_mode`: `{answer.fault_injection_mode}`",
        f"- `demo_fault_count`: `{answer.fault_injection_count}`",
    ]
    if answer.answer_depth == "detailed":
        lines.append("- `depth_note`: `detailed выбран автоматически по обзорной формулировке вопроса`")
    if _selected_draft_strategy() == "demo" or _selected_verification_sensitivity() == "demo":
        lines.append("- `demo_note`: `Demo mode active: higher rewrite sensitivity for conference visualization`")
    if answer.fault_injection_active and answer.fault_injection_summary:
        lines.append(f"- `fault_note`: `{answer.fault_injection_summary}`")
    if answer.citations:
        lines.extend(["", "### Citations", ""])
        for citation in answer.citations:
            lines.append(
                f"- [{citation.evidence_id}] {citation.file_name} / {citation.section_title}: {citation.support}"
            )
    if answer.omitted_points:
        lines.extend(["", "### Omitted Points", ""])
        for item in answer.omitted_points:
            lines.append(f"- {item}")
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start() -> None:
    config = _base_config()
    cl.user_session.set(SESSION_DRAFT_MODEL, config.draft_model_name)
    cl.user_session.set(SESSION_VERIFICATION_MODEL, config.verification_model_name)
    cl.user_session.set(SESSION_DRAFT_STRATEGY, config.draft_strategy)
    cl.user_session.set(SESSION_VERIFICATION_SENSITIVITY, config.verification_sensitivity)
    cl.user_session.set(SESSION_DEMO_FAULT_MODE, config.demo_fault_mode)
    cl.user_session.set(SESSION_DEMO_FAULT_COUNT, config.demo_fault_count)
    await cl.ChatSettings(
        [
            Select(
                id=SETTINGS_DRAFT_MODEL,
                label="Draft model",
                values=list(config.available_draft_models),
                initial_value=config.draft_model_name,
                tooltip="Локальная модель LM Studio для первичного ответа.",
            ),
            Select(
                id=SETTINGS_VERIFICATION_MODEL,
                label="Verification model",
                values=list(config.available_verification_models),
                initial_value=config.verification_model_name,
                tooltip="Локальная модель LM Studio для claims, verification и rewrite.",
            ),
            Select(
                id=SETTINGS_DRAFT_STRATEGY,
                label="Draft strategy",
                values=["conservative", "balanced", "demo"],
                initial_value=config.draft_strategy,
                tooltip="Demo повышает шанс обзорного draft с несколькими проверяемыми claims.",
            ),
            Select(
                id=SETTINGS_VERIFICATION_SENSITIVITY,
                label="Verification sensitivity",
                values=["conservative", "balanced", "demo"],
                initial_value=config.verification_sensitivity,
                tooltip="Demo строже относится к partial claims и чаще запускает локальный rewrite.",
            ),
            Select(
                id=SETTINGS_DEMO_FAULT_MODE,
                label="Verification demo",
                values=["off", "deterministic"],
                initial_value=config.demo_fault_mode,
                tooltip="Controlled mismatch demo intentionally perturbs 1-2 claims for visualization.",
            ),
            Select(
                id=SETTINGS_DEMO_FAULT_COUNT,
                label="Fault count",
                values=["1", "2"],
                initial_value=str(config.demo_fault_count),
                tooltip="Сколько контролируемых неточностей внести в draft в demo-режиме.",
            ),
        ]
    ).send()
    await cl.Message(
        content=(
            "VeriFlow RAG demo готов. Задайте вопрос, и я сначала построю `draft answer`, "
            "а затем по кнопке `Агентная проверка` можно будет запустить проверку claims. "
            f"Draft model: `{config.draft_model_name}`, verification model: `{config.verification_model_name}`."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    config = _base_config()
    try:
        resolved = (
            config.with_draft_model(str(settings.get(SETTINGS_DRAFT_MODEL, config.draft_model_name)))
            .with_verification_model(
                str(settings.get(SETTINGS_VERIFICATION_MODEL, config.verification_model_name))
            )
            .with_draft_strategy(str(settings.get(SETTINGS_DRAFT_STRATEGY, config.draft_strategy)))
            .with_verification_sensitivity(
                str(settings.get(SETTINGS_VERIFICATION_SENSITIVITY, config.verification_sensitivity))
            )
            .with_demo_fault_mode(str(settings.get(SETTINGS_DEMO_FAULT_MODE, config.demo_fault_mode)))
            .with_demo_fault_count(int(settings.get(SETTINGS_DEMO_FAULT_COUNT, config.demo_fault_count)))
        )
    except ValueError as exc:
        await cl.Message(content=f"❌ {exc}").send()
        return
    cl.user_session.set(SESSION_DRAFT_MODEL, resolved.draft_model_name)
    cl.user_session.set(SESSION_VERIFICATION_MODEL, resolved.verification_model_name)
    cl.user_session.set(SESSION_DRAFT_STRATEGY, resolved.draft_strategy)
    cl.user_session.set(SESSION_VERIFICATION_SENSITIVITY, resolved.verification_sensitivity)
    cl.user_session.set(SESSION_DEMO_FAULT_MODE, resolved.demo_fault_mode)
    cl.user_session.set(SESSION_DEMO_FAULT_COUNT, resolved.demo_fault_count)
    await cl.Message(
        content=(
            "⚙️ Настройки обновлены: "
            f"`draft_model={resolved.draft_model_name}`, "
            f"`verification_model={resolved.verification_model_name}`, "
            f"`draft_strategy={resolved.draft_strategy}`, "
            f"`verification_sensitivity={resolved.verification_sensitivity}`, "
            f"`demo_fault_mode={resolved.demo_fault_mode}`, "
            f"`demo_fault_count={resolved.demo_fault_count}`."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = str(message.content).strip()
    if not query:
        await cl.Message(content="Вопрос пустой. Нужен текстовый запрос.").send()
        return

    config = _selected_config()
    synthesis_service = build_synthesis_service(config)
    status = cl.Message(
        content=(
            f"Ищу evidence и собираю первичный ответ через `{config.draft_model_name}` "
            f"(`draft_strategy={config.draft_strategy}`)..."
        )
    )
    await status.send()

    try:
        bundle = await asyncio.to_thread(synthesis_service.run_query, query)
    except LMStudioClientError as exc:
        status.content = f"❌ {exc}"
        await status.update()
        return

    cl.user_session.set(SESSION_DRAFT_BUNDLE, bundle)
    cl.user_session.set(SESSION_VERIFICATION_RUNNING, False)

    actions = []
    if not bundle.synthesized_answer.insufficient_context:
        actions.append(
            cl.Action(
                name="run_agent_verification",
                payload={"query": query},
                label="Агентная проверка",
                tooltip="Запустить декомпозицию claims и перепроверку ответа",
                icon="bot",
            )
        )

    draft_message = cl.Message(
        content=_render_draft_markdown(bundle),
        actions=actions or None,
    )
    await draft_message.send()

    status.content = "✅ Черновой ответ готов."
    await status.update()


@cl.action_callback("run_agent_verification")
async def run_agent_verification(action: cl.Action) -> None:
    if cl.user_session.get(SESSION_VERIFICATION_RUNNING):
        await cl.Message(content="Проверка уже выполняется для текущего ответа.").send()
        return

    draft_bundle: SynthesisResultBundle | None = cl.user_session.get(SESSION_DRAFT_BUNDLE)
    if draft_bundle is None:
        await cl.Message(content="Не найден draft answer для проверки. Сначала задайте вопрос.").send()
        return

    cl.user_session.set(SESSION_VERIFICATION_RUNNING, True)
    config = _selected_config()
    orchestrator = build_verification_orchestrator(config)
    verification_profile = select_verification_profile(config)
    progress_lines = [
        "## Агентная проверка",
        "",
        f"Подготовка verification pipeline через `{config.verification_model_name}`...",
        "",
        f"- `verification_sensitivity`: `{verification_profile.sensitivity}`",
    ]
    if config.draft_strategy == "demo" or config.verification_sensitivity == "demo":
        progress_lines.append(
            "- `demo_note`: `Demo mode active: higher rewrite sensitivity for conference visualization`"
        )

    progress = cl.Message(content="\n".join(progress_lines))
    highlighted = cl.Message(content="## Highlighted Answer\n\n_Проверка ещё не началась._")
    claims_msg = cl.Message(content="## Claims\n\n_Claims ещё не извлечены._")
    final_msg = cl.Message(content="## Final Answer\n\n_Финальный ответ появится после проверки._")
    await progress.send()
    await highlighted.send()
    await claims_msg.send()
    await final_msg.send()

    try:
        draft_answer = draft_bundle.synthesized_answer.answer

        progress.content = "## Агентная проверка\n\nИзвлечение claims..."
        await progress.update()
        claims = await asyncio.to_thread(orchestrator.claim_extractor.extract_claims, draft_answer)
        claims = merge_injected_claims(
            draft_answer=draft_answer,
            claims=claims,
            injected_spans=draft_bundle.synthesized_answer.fault_injection_spans,
        )
        claims_msg.content = render_claim_table([])
        await claims_msg.update()

        claim_results = []
        claim_evidence_map = {}
        injected_spans = draft_bundle.synthesized_answer.fault_injection_spans
        for claim in claims:
            progress.content = f"## Агентная проверка\n\nПроверка `{claim.claim_id}`..."
            await progress.update()

            evidence_blocks = await asyncio.to_thread(
                orchestrator.retrieval_service.retrieve_for_claim,
                claim.claim_text,
            )
            prepared = orchestrator.retrieval_service.prepare_claim_evidence(evidence_blocks)
            claim_evidence_map[claim.claim_id] = prepared
            result = await asyncio.to_thread(
                orchestrator.claim_verifier.verify_claim,
                claim,
                prepared,
            )
            result = apply_demo_verification_override(
                result,
                draft_bundle.synthesized_answer.fault_injection_spans,
            )
            result.rewrite_source_span = select_rewrite_span(draft_answer, result)
            if (
                injected_spans
                and result.status != "supported"
                and any(
                    _normalize_span(injected.injected_span) == _normalize_span(result.source_span)
                    or _normalize_span(injected.injected_span) == _normalize_span(result.claim_text)
                    or _normalize_span(injected.injected_span) == _normalize_span(result.rewrite_source_span)
                    for injected in injected_spans
                )
            ):
                result.rewrite_needed = True
            claim_results.append(result)

            highlighted.content = "## Highlighted Answer\n\n" + render_highlighted_answer(
                draft_answer,
                claim_results,
            )
            claims_msg.content = render_claim_table(claim_results)
            await highlighted.update()
            await claims_msg.update()

        decision = should_trigger_rewrite(
            draft_answer=draft_answer,
            claim_results=claim_results,
            partial_ratio_threshold=verification_profile.partial_ratio_threshold,
            problem_span_ratio_threshold=verification_profile.problem_span_ratio_threshold,
        )

        rewritten_spans: dict[str, str] = {}
        force_rewrite = any(
            result.status != "supported"
            and any(
                _normalize_span(injected.injected_span) == _normalize_span(result.source_span)
                or _normalize_span(injected.injected_span) == _normalize_span(result.rewrite_source_span)
                for injected in injected_spans
            )
            for result in claim_results
        )

        if decision.rewrite_triggered or force_rewrite:
            progress.content = "## Агентная проверка\n\nПереписывание проблемных фрагментов..."
            await progress.update()
            for result in claim_results:
                if not result.rewrite_needed or result.status == "supported":
                    continue
                rewritten = await asyncio.to_thread(
                    orchestrator.claim_rewriter.rewrite_claim,
                    draft_answer=draft_answer,
                    claim_result=result,
                    evidence_blocks=claim_evidence_map.get(result.claim_id, []),
                )
                if rewritten:
                    rewritten_spans[result.claim_id] = rewritten

        final_answer, applied_rewrites = apply_rewrites(
            draft_answer=draft_answer,
            claim_results=claim_results,
            rewritten_spans=rewritten_spans,
        )

        final_parts = ["## Final Answer", "", final_answer, "", render_rewrite_diff(applied_rewrites)]
        final_msg.content = "\n".join(final_parts)
        await final_msg.update()

        progress.content = "## Агентная проверка\n\n✅ Проверка завершена."
        await progress.update()
    except LMStudioClientError as exc:
        progress.content = f"## Агентная проверка\n\n❌ {exc}"
        await progress.update()
    finally:
        cl.user_session.set(SESSION_VERIFICATION_RUNNING, False)
