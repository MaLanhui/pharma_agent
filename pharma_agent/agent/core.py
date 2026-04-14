from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from openai import OpenAI
from rdkit import Chem, rdBase

from pharma_agent.agent.tools import evaluate_molecule, evaluate_molecule_batch, query_rules, search_pubmed
from pharma_agent.config import settings
from pharma_agent.mol.evaluator import preserves_core, scaffold_similarity


GENERATOR_PROMPT = """You are a medicinal chemistry optimization assistant.

Requirements:
1. Output JSON only.
2. Each candidate must be a valid small-molecule SMILES.
3. Preserve the provided core scaffold as much as possible.
4. Favor better oral ADME, lower medicinal chemistry risk, and better toxicity profile.
5. Avoid previously tried SMILES.

JSON schema:
{
  "candidates": [
    {
      "smiles": "string",
      "modification": "string",
      "rationale": "string",
      "expected_gain": "string"
    }
  ]
}
"""


REPORT_PROMPT = """You are a medicinal chemistry reporting assistant.

Requirements:
1. Use only the provided evidence.
2. Each suggestion should cite at least one paper citation when available.
3. Output JSON only.

JSON schema:
{
  "target_summary": "string",
  "optimization_summary": "string",
  "molecule_assessment": "string",
  "overall_grade": "A/B+/B/C+/C/D/N/A",
  "overall_comment": "string",
  "suggestions": [
    {
      "headline": "string",
      "recommendation": "string",
      "paper_citations": ["P1"],
      "rule_citations": ["R1"]
    }
  ]
}
"""


def run_auto_optimization(
    target: str,
    lead_smiles: str,
    *,
    goal_score: float = 82.0,
    max_risk_score: float | None = 42.0,
    min_toxicity_score: float | None = 58.0,
    min_toxicity_class: int | None = 4,
    min_bioavailability_score: float | None = 0.45,
    max_iterations: int = 4,
    candidate_count: int = 4,
    pubmed_max: int = 4,
    top_k: int = 4,
) -> dict[str, Any]:
    session = _initialize_session(
        target=target,
        lead_smiles=lead_smiles,
        mode="auto",
        goal_score=goal_score,
        max_risk_score=max_risk_score,
        min_toxicity_score=min_toxicity_score,
        min_toxicity_class=min_toxicity_class,
        min_bioavailability_score=min_bioavailability_score,
        max_iterations=max_iterations,
        candidate_count=candidate_count,
        pubmed_max=pubmed_max,
        top_k=top_k,
    )
    while not session["stopped"] and session["current_iteration"] < session["config"]["max_iterations"]:
        session = _run_iteration(session, selected_smiles=session["current_seed_smiles"], auto_select=True)
    return _build_result(session)


def start_manual_optimization(
    target: str,
    lead_smiles: str,
    *,
    goal_score: float = 82.0,
    max_risk_score: float | None = 42.0,
    min_toxicity_score: float | None = 58.0,
    min_toxicity_class: int | None = 4,
    min_bioavailability_score: float | None = 0.45,
    max_iterations: int = 4,
    candidate_count: int = 4,
    pubmed_max: int = 4,
    top_k: int = 4,
) -> dict[str, Any]:
    session = _initialize_session(
        target=target,
        lead_smiles=lead_smiles,
        mode="manual",
        goal_score=goal_score,
        max_risk_score=max_risk_score,
        min_toxicity_score=min_toxicity_score,
        min_toxicity_class=min_toxicity_class,
        min_bioavailability_score=min_bioavailability_score,
        max_iterations=max_iterations,
        candidate_count=candidate_count,
        pubmed_max=pubmed_max,
        top_k=top_k,
    )
    session = _run_iteration(session, selected_smiles=session["current_seed_smiles"], auto_select=False)
    return _build_result(session)


def continue_manual_optimization(session: dict[str, Any], selected_smiles: str) -> dict[str, Any]:
    if session["stopped"]:
        return _build_result(session)
    if session["mode"] != "manual":
        raise ValueError("Manual continuation can only be used with manual mode sessions.")
    session = _run_iteration(session, selected_smiles=selected_smiles, auto_select=False)
    return _build_result(session)


def summarize_session(session: dict[str, Any]) -> dict[str, Any]:
    return _build_result(session)


def _initialize_session(
    *,
    target: str,
    lead_smiles: str,
    mode: str,
    goal_score: float,
    max_risk_score: float | None,
    min_toxicity_score: float | None,
    min_toxicity_class: int | None,
    min_bioavailability_score: float | None,
    max_iterations: int,
    candidate_count: int,
    pubmed_max: int,
    top_k: int,
) -> dict[str, Any]:
    lead_evaluation = evaluate_molecule(lead_smiles)
    if not lead_evaluation.get("valid"):
        raise ValueError("Initial SMILES is invalid and cannot be optimized.")

    config = {
        "goal_score": float(goal_score),
        "max_risk_score": max_risk_score,
        "min_toxicity_score": min_toxicity_score,
        "min_toxicity_class": min_toxicity_class,
        "min_bioavailability_score": min_bioavailability_score,
        "max_iterations": int(max_iterations),
        "candidate_count": int(candidate_count),
        "pubmed_max": int(pubmed_max),
        "top_k": int(top_k),
    }
    session = {
        "target": target,
        "mode": mode,
        "config": config,
        "steps": [],
        "iterations": [],
        "history_points": [],
        "tried_smiles": [lead_evaluation["canonical_smiles"]],
        "latest_candidates": [],
        "current_seed_smiles": lead_evaluation["canonical_smiles"],
        "current_iteration": 0,
        "stopped": False,
        "stop_reason": None,
        "recommended_next_smiles": lead_evaluation["canonical_smiles"],
        "seed_record": None,
    }

    session["evidence"] = _collect_evidence(target, pubmed_max=pubmed_max, top_k=top_k, session=session)
    seed_record = _build_record(
        name="Seed",
        smiles=lead_evaluation["canonical_smiles"],
        iteration=0,
        source="user_seed",
        modification="User-provided seed molecule",
        rationale="Baseline starting point for optimization.",
        expected_gain="Establish a baseline profile before iterative optimization.",
        seed_smiles=lead_evaluation["canonical_smiles"],
        evaluation=lead_evaluation,
        config=config,
    )
    session["seed_record"] = seed_record
    session["best_record"] = seed_record
    session["history_points"].append(
        {
            "label": "Seed",
            "current_score": seed_record["overall_score"],
            "best_score": seed_record["overall_score"],
            "risk_score": seed_record["risk_score"],
            "smiles": seed_record["smiles"],
        }
    )
    _remember(
        session,
        step="Step 0",
        summary="Evaluated the initial lead molecule.",
        tool="evaluate_molecule",
        tool_input={"smiles": lead_smiles},
        tool_output=_compact_eval(lead_evaluation),
        payload=_compact_record(seed_record),
    )
    if seed_record["goal_status"]["passed"]:
        session["stopped"] = True
        session["stop_reason"] = "Initial molecule already satisfies the configured optimization goals."
    return session


def _run_iteration(session: dict[str, Any], *, selected_smiles: str, auto_select: bool) -> dict[str, Any]:
    if session["stopped"]:
        return session

    next_iteration = session["current_iteration"] + 1
    if next_iteration > session["config"]["max_iterations"]:
        session["stopped"] = True
        session["stop_reason"] = "Reached the maximum iteration limit."
        return session

    candidate_specs = _generate_candidates(
        target=session["target"],
        seed_smiles=selected_smiles,
        evidence=session["evidence"],
        best_record=session["best_record"],
        tried_smiles=session["tried_smiles"],
        config=session["config"],
        candidate_count=session["config"]["candidate_count"],
        iteration_index=next_iteration,
    )
    _remember(
        session,
        step=f"Step {next_iteration}.1",
        summary=f"Generated {len(candidate_specs)} raw candidate molecules.",
        tool="generate_candidates",
        tool_input={"seed_smiles": selected_smiles, "iteration": next_iteration},
        tool_output={"count": len(candidate_specs), "smiles": [item["smiles"] for item in candidate_specs]},
        payload=candidate_specs,
    )
    if not candidate_specs:
        session["stopped"] = True
        session["stop_reason"] = "No valid candidates were generated."
        return session

    start = perf_counter()
    evaluations = evaluate_molecule_batch(
        [item["smiles"] for item in candidate_specs],
        refine_top_n=settings.protox3_refine_top_n,
    )
    duration_ms = int((perf_counter() - start) * 1000)
    candidate_records = _merge_candidate_evaluations(
        candidate_specs=candidate_specs,
        evaluations=evaluations,
        seed_smiles=selected_smiles,
        iteration_index=next_iteration,
        config=session["config"],
    )
    _remember(
        session,
        step=f"Step {next_iteration}.2",
        summary="Completed batch evaluation for this round.",
        tool="evaluate_molecule_batch",
        tool_input={"count": len(candidate_specs)},
        tool_output={"count": len(candidate_records), "top_score": candidate_records[0]["overall_score"] if candidate_records else None},
        duration_ms=duration_ms,
        payload=[_compact_record(item) for item in candidate_records[:3]],
    )
    if not candidate_records:
        session["stopped"] = True
        session["stop_reason"] = "All candidates failed structural validation or core-preservation checks."
        return session

    round_best = candidate_records[0]
    session["latest_candidates"] = candidate_records
    session["iterations"].append(
        {
            "iteration": next_iteration,
            "seed_smiles": selected_smiles,
            "candidate_count": len(candidate_records),
            "best_candidate": _compact_record(round_best),
            "candidates": candidate_records,
        }
    )
    session["current_iteration"] = next_iteration
    session["history_points"].append(
        {
            "label": f"Iter {next_iteration}",
            "current_score": round_best["overall_score"],
            "best_score": max(session["best_record"]["overall_score"], round_best["overall_score"]),
            "risk_score": round_best["risk_score"],
            "smiles": round_best["smiles"],
        }
    )
    for record in candidate_records:
        if record["smiles"] not in session["tried_smiles"]:
            session["tried_smiles"].append(record["smiles"])
    if _record_sort_key(round_best, session["config"]) < _record_sort_key(session["best_record"], session["config"]):
        session["best_record"] = round_best

    session["recommended_next_smiles"] = round_best["smiles"]
    if auto_select:
        session["current_seed_smiles"] = round_best["smiles"]

    if session["best_record"]["goal_status"]["passed"]:
        session["stopped"] = True
        session["stop_reason"] = "Best molecule satisfies all configured optimization goals."
    elif session["current_iteration"] >= session["config"]["max_iterations"]:
        session["stopped"] = True
        session["stop_reason"] = "Reached the maximum iteration limit."
    elif auto_select and round_best["smiles"] == selected_smiles:
        session["stopped"] = True
        session["stop_reason"] = "Automatic mode could not find a better next seed and stopped early."
    return session


def _collect_evidence(target: str, *, pubmed_max: int, top_k: int, session: dict[str, Any]) -> dict[str, Any]:
    papers = _merge_papers(
        _timed_search(session, "Step E1", f"{target} inhibitor medicinal chemistry OR small molecule drug design", pubmed_max),
        _timed_search(session, "Step E2", f"{target} resistance OR selectivity optimization medicinal chemistry", max(2, pubmed_max - 1)),
    )[: max(pubmed_max + 1, 4)]
    rules = _merge_rules(
        _timed_retrieve(session, "Step E3", f"{target} kinase inhibitor selectivity resistance optimization", top_k),
        _timed_retrieve(session, "Step E4", f"{target} oral exposure ADMET Lipinski Veber optimization", top_k),
    )[: max(top_k + 1, 4)]
    return _build_evidence_bundle(papers, rules)


def _timed_search(session: dict[str, Any], step: str, query: str, retmax: int) -> list[dict[str, Any]]:
    start = perf_counter()
    try:
        results = search_pubmed(query, retmax=retmax)
        _remember(
            session,
            step=step,
            summary=f"PubMed returned {len(results)} paper summaries.",
            tool="search_pubmed",
            tool_input={"query": query, "retmax": retmax},
            tool_output={"count": len(results), "pmids": [item["pmid"] for item in results]},
            duration_ms=int((perf_counter() - start) * 1000),
            payload=results,
        )
        return results
    except Exception as exc:
        _remember(
            session,
            step=step,
            summary="PubMed retrieval failed; continuing with the remaining evidence.",
            tool="search_pubmed",
            tool_input={"query": query, "retmax": retmax},
            tool_output={"error": str(exc)},
            duration_ms=int((perf_counter() - start) * 1000),
            status="failed",
        )
        return []


def _timed_retrieve(session: dict[str, Any], step: str, question: str, top_k: int) -> list[dict[str, Any]]:
    start = perf_counter()
    try:
        results = query_rules(question, top_k=top_k)
        _remember(
            session,
            step=step,
            summary=f"Local rule retrieval returned {len(results)} snippets.",
            tool="query_rules",
            tool_input={"question": question, "top_k": top_k},
            tool_output={"count": len(results), "scores": [item["score"] for item in results]},
            duration_ms=int((perf_counter() - start) * 1000),
            payload=results,
        )
        return results
    except Exception as exc:
        _remember(
            session,
            step=step,
            summary="Local rule retrieval failed; continuing without those snippets.",
            tool="query_rules",
            tool_input={"question": question, "top_k": top_k},
            tool_output={"error": str(exc)},
            duration_ms=int((perf_counter() - start) * 1000),
            status="failed",
        )
        return []


def _generate_candidates(
    *,
    target: str,
    seed_smiles: str,
    evidence: dict[str, Any],
    best_record: dict[str, Any],
    tried_smiles: list[str],
    config: dict[str, Any],
    candidate_count: int,
    iteration_index: int,
) -> list[dict[str, Any]]:
    seed_evaluation = evaluate_molecule(seed_smiles)
    payload = {
        "target": target,
        "iteration_index": iteration_index,
        "candidate_count": candidate_count,
        "seed_smiles": seed_smiles,
        "seed_scaffold": seed_evaluation.get("core_scaffold_smiles") or "",
        "seed_metrics": _compact_eval(seed_evaluation),
        "best_so_far": _compact_record(best_record),
        "goal_constraints": _goal_payload(config),
        "already_tried_smiles": tried_smiles[-20:],
        "evidence": {
            "papers": [{"evidence_id": p["evidence_id"], "title": p["title"], "pmid": p["pmid"]} for p in evidence["papers"][:3]],
            "rules": [{"evidence_id": r["evidence_id"], "text": r["text"][:180]} for r in evidence["rules"][:3]],
        },
    }
    if not settings.deepseek_api_key:
        raw_candidates = _fallback_candidates(seed_smiles)
    else:
        client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            temperature=0.45,
            messages=[
                {"role": "system", "content": GENERATOR_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
        )
        raw_candidates = _parse_json(response.choices[0].message.content or "").get("candidates", [])

    deduped: list[dict[str, Any]] = []
    seen = set(tried_smiles)
    for item in raw_candidates:
        smiles = str(item.get("smiles", "")).strip()
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen or not preserves_core(seed_smiles, canonical):
            continue
        seen.add(canonical)
        deduped.append(
            {
                "smiles": canonical,
                "modification": str(item.get("modification", "")).strip() or "Peripheral substitution refinement",
                "rationale": str(item.get("rationale", "")).strip() or "Attempt to improve the multi-parameter profile.",
                "expected_gain": str(item.get("expected_gain", "")).strip() or "Improve optimization goals while preserving the core scaffold.",
            }
        )
        if len(deduped) >= candidate_count:
            break
    return deduped or _fallback_candidates(seed_smiles)


def _merge_candidate_evaluations(
    *,
    candidate_specs: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    seed_smiles: str,
    iteration_index: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for spec, evaluation in zip(candidate_specs, evaluations):
        if not evaluation.get("valid"):
            continue
        merged.append(
            _build_record(
                name=f"Iter {iteration_index} Candidate {len(merged) + 1}",
                smiles=evaluation["canonical_smiles"],
                iteration=iteration_index,
                source="generated",
                modification=spec["modification"],
                rationale=spec["rationale"],
                expected_gain=spec["expected_gain"],
                seed_smiles=seed_smiles,
                evaluation=evaluation,
                config=config,
            )
        )
    merged.sort(key=lambda item: _record_sort_key(item, config))
    return merged


def _build_record(
    *,
    name: str,
    smiles: str,
    iteration: int,
    source: str,
    modification: str,
    rationale: str,
    expected_gain: str,
    seed_smiles: str,
    evaluation: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "name": name,
        "smiles": smiles,
        "iteration": iteration,
        "source": source,
        "modification": modification,
        "rationale": rationale,
        "expected_gain": expected_gain,
        "similarity_to_seed": scaffold_similarity(seed_smiles, smiles),
        "core_preserved": preserves_core(seed_smiles, smiles),
        "evaluation": evaluation,
        "overall_score": evaluation["overall_score"],
        "risk_score": evaluation["risk_score"],
    }
    record["goal_status"] = _goal_status(evaluation, config)
    return record


def _record_sort_key(record: dict[str, Any], config: dict[str, Any]) -> tuple[Any, ...]:
    goal_status = record.get("goal_status") or _goal_status(record["evaluation"], config)
    tox_score = record["evaluation"].get("toxicity", {}).get("toxicity_score")
    return (
        0 if goal_status["passed"] else 1,
        -goal_status["met_count"],
        -record["overall_score"],
        record["risk_score"],
        -(tox_score if tox_score is not None else -1.0),
        -record["similarity_to_seed"],
    )


def _goal_status(evaluation: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    toxicity = evaluation.get("toxicity", {})
    items = []
    specs = [
        ("overall_score", "Overall Score", config.get("goal_score"), evaluation.get("overall_score"), "min"),
        ("risk_score", "Risk Score", config.get("max_risk_score"), evaluation.get("risk_score"), "max"),
        ("toxicity_score", "ProTox3 Score", config.get("min_toxicity_score"), toxicity.get("toxicity_score"), "min"),
        ("toxicity_class", "Toxicity Class", config.get("min_toxicity_class"), toxicity.get("predicted_tox_class"), "min"),
        ("bioavailability", "Bioavailability", config.get("min_bioavailability_score"), evaluation.get("bioavailability_score"), "min"),
    ]
    for key, label, target, actual, mode in specs:
        enabled = target is not None and target != 0
        passed = not enabled
        if enabled and actual is not None:
            passed = actual >= target if mode == "min" else actual <= target
        elif enabled:
            passed = False
        items.append({"key": key, "label": label, "mode": mode, "target": target, "actual": actual, "enabled": enabled, "passed": passed})
    enabled_items = [item for item in items if item["enabled"]]
    return {
        "passed": all(item["passed"] for item in enabled_items) if enabled_items else True,
        "met_count": sum(1 for item in enabled_items if item["passed"]),
        "total_count": len(enabled_items),
        "items": items,
    }


def _goal_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal_score": config.get("goal_score"),
        "max_risk_score": config.get("max_risk_score"),
        "min_toxicity_score": config.get("min_toxicity_score"),
        "min_toxicity_class": config.get("min_toxicity_class"),
        "min_bioavailability_score": config.get("min_bioavailability_score"),
    }


def _build_result(session: dict[str, Any]) -> dict[str, Any]:
    best_record = session["best_record"]
    return {
        "mode": session["mode"],
        "target": session["target"],
        "config": session["config"],
        "report": _synthesize_report(session),
        "seed_record": session["seed_record"],
        "best_record": best_record,
        "best_molecule": best_record["evaluation"],
        "goal_status": best_record["goal_status"],
        "seed_goal_status": session["seed_record"]["goal_status"],
        "steps": session["steps"],
        "iterations": session["iterations"],
        "history": session["history_points"],
        "papers": session["evidence"]["papers"],
        "rules": session["evidence"]["rules"],
        "latest_candidates": session["latest_candidates"],
        "stopped": session["stopped"],
        "stop_reason": session["stop_reason"],
        "recommended_next_smiles": session.get("recommended_next_smiles"),
        "session": session,
    }


def _synthesize_report(session: dict[str, Any]) -> dict[str, Any]:
    best_record = session["best_record"]
    evidence = session["evidence"]
    payload = {
        "target": session["target"],
        "goals": _goal_payload(session["config"]),
        "best_molecule": {
            "smiles": best_record["smiles"],
            "overall_score": best_record["overall_score"],
            "overall_grade": best_record["evaluation"]["overall_grade"],
            "goal_status": best_record["goal_status"],
            "evaluation": _compact_eval(best_record["evaluation"]),
            "modification": best_record["modification"],
            "rationale": best_record["rationale"],
        },
        "iterations": [{"iteration": item["iteration"], "seed_smiles": item["seed_smiles"], "best_candidate": item["best_candidate"]} for item in session["iterations"]],
        "pubmed": evidence["papers"],
        "rules": evidence["rules"],
        "stop_reason": session["stop_reason"],
    }
    if not settings.deepseek_api_key:
        return _fallback_report(session, mode="fallback-no-api")

    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": REPORT_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
        )
        parsed = _parse_json(response.choices[0].message.content or "")
        normalized = _normalize_report(parsed, evidence_bundle=evidence)
        normalized["generation_mode"] = "deepseek"
        return normalized
    except Exception:
        return _fallback_report(session, mode="fallback-on-error")


def _fallback_report(session: dict[str, Any], mode: str) -> dict[str, Any]:
    best_record = session["best_record"]
    evaluation = best_record["evaluation"]
    tox = evaluation.get("toxicity", {})
    evidence = session["evidence"]
    paper_ids = [paper["evidence_id"] for paper in evidence["papers"]]
    rule_ids = [rule["evidence_id"] for rule in evidence["rules"]]
    report = {
        "target_summary": _fallback_target_summary(session["target"], evidence["papers"]),
        "optimization_summary": f"Completed {session['current_iteration']} optimization rounds. Stop reason: {session['stop_reason'] or 'not stopped'}.",
        "molecule_assessment": (
            f"Best molecule score={evaluation['overall_score']}, grade={evaluation['overall_grade']}, "
            f"risk={evaluation['risk_score']}, bioavailability={evaluation.get('bioavailability_score')}, "
            f"toxicity_score={tox.get('toxicity_score')}."
        ),
        "overall_grade": evaluation["overall_grade"],
        "overall_comment": f"Best molecule came from iteration {best_record['iteration']} with direction: {best_record['modification']}.",
        "suggestions": [
            {
                "headline": "Keep optimizing around the current best scaffold",
                "recommendation": "Prioritize peripheral changes that preserve the scaffold and improve unmet goal constraints.",
                "paper_citations": paper_ids[:1],
                "rule_citations": rule_ids[:1],
            },
            {
                "headline": "Focus the next round on the weakest metric",
                "recommendation": "Use risk, toxicity score, and bioavailability gaps to steer the next design move instead of optimizing score alone.",
                "paper_citations": paper_ids[1:2] or paper_ids[:1],
                "rule_citations": rule_ids[1:2] or rule_ids[:1],
            },
        ],
        "generation_mode": mode,
    }
    return _normalize_report(report, evidence_bundle=evidence)


def _normalize_report(raw_report: dict[str, Any], *, evidence_bundle: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    valid_paper_ids = {paper["evidence_id"] for paper in evidence_bundle["papers"]}
    valid_rule_ids = {rule["evidence_id"] for rule in evidence_bundle["rules"]}
    paper_number_map = {paper["evidence_id"]: paper["citation_number"] for paper in evidence_bundle["papers"]}
    suggestions = []
    for item in raw_report.get("suggestions", [])[:3]:
        paper_citations = [cid for cid in item.get("paper_citations", []) if cid in valid_paper_ids]
        rule_citations = [cid for cid in item.get("rule_citations", []) if cid in valid_rule_ids]
        if valid_paper_ids and not paper_citations:
            paper_citations = [sorted(valid_paper_ids)[0]]
        suggestions.append(
            {
                "headline": str(item.get("headline", "Optimization suggestion")).strip() or "Optimization suggestion",
                "recommendation": str(item.get("recommendation", "")).strip() or "No recommendation was generated.",
                "paper_citations": paper_citations,
                "rule_citations": rule_citations,
                "citation_numbers": [paper_number_map[cid] for cid in paper_citations if cid in paper_number_map],
            }
        )
    return {
        "target_summary": str(raw_report.get("target_summary", "")).strip() or "No target summary generated.",
        "optimization_summary": str(raw_report.get("optimization_summary", "")).strip() or "No optimization summary generated.",
        "molecule_assessment": str(raw_report.get("molecule_assessment", "")).strip() or "No molecule assessment generated.",
        "overall_grade": str(raw_report.get("overall_grade", "N/A")).strip() or "N/A",
        "overall_comment": str(raw_report.get("overall_comment", "")).strip() or "Evidence is limited.",
        "suggestions": suggestions,
        "paper_citations": evidence_bundle["papers"],
        "rule_citations": evidence_bundle["rules"],
        "generation_mode": raw_report.get("generation_mode", "normalized"),
    }


def _build_evidence_bundle(papers: list[dict[str, Any]], rules: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    paper_bundle = []
    rule_bundle = []
    for idx, paper in enumerate(papers, start=1):
        item = dict(paper)
        item["evidence_id"] = f"P{idx}"
        item["citation_number"] = idx
        paper_bundle.append(item)
    for idx, rule in enumerate(rules, start=1):
        item = dict(rule)
        item["evidence_id"] = f"R{idx}"
        item["citation_number"] = idx
        rule_bundle.append(item)
    return {"papers": paper_bundle, "rules": rule_bundle}


def _merge_papers(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            pmid = item.get("pmid")
            if not pmid or pmid in seen:
                continue
            seen.add(pmid)
            merged.append(item)
    return merged


def _merge_rules(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            key = item.get("id") or item.get("text")
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _fallback_target_summary(target: str, papers: list[dict[str, Any]]) -> str:
    if not papers:
        return f"Evidence for {target} is limited in the current retrieval set."
    titles = "; ".join(paper["title"] for paper in papers[:3] if paper.get("title"))
    return f"Retrieved literature around {target} mainly focuses on activity, selectivity, resistance, and oral ADME balance. Representative topics: {titles}."


def _remember(
    session: dict[str, Any],
    *,
    step: str,
    summary: str,
    tool: str,
    tool_input: Any,
    tool_output: Any,
    status: str = "completed",
    duration_ms: int | None = None,
    payload: Any = None,
) -> None:
    session["steps"].append(
        {
            "step": step,
            "summary": summary,
            "tool": tool,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "status": status,
            "duration_ms": duration_ms,
            "payload": payload,
        }
    )


def _parse_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback_candidates(seed_smiles: str) -> list[dict[str, Any]]:
    return [
        {
            "smiles": seed_smiles,
            "modification": "Fallback to the current seed",
            "rationale": "No reliable new candidate was generated, so the workflow keeps the current seed.",
            "expected_gain": "Maintain continuity instead of breaking the workflow with invalid candidates.",
        }
    ]


def _compact_eval(evaluation: dict[str, Any]) -> dict[str, Any]:
    if not evaluation.get("valid"):
        return {"valid": False, "error": evaluation.get("error")}
    toxicity = evaluation.get("toxicity", {})
    return {
        "smiles": evaluation.get("canonical_smiles"),
        "overall_score": evaluation.get("overall_score"),
        "overall_grade": evaluation.get("overall_grade"),
        "base_overall_score": evaluation.get("base_overall_score"),
        "risk_score": evaluation.get("risk_score"),
        "bioavailability_score": evaluation.get("bioavailability_score"),
        "gi_absorption": evaluation.get("gi_absorption"),
        "logp": evaluation.get("logp"),
        "tpsa": evaluation.get("tpsa"),
        "sa": evaluation.get("sa_proxy"),
        "toxicity_score": toxicity.get("toxicity_score"),
        "toxicity_class": toxicity.get("predicted_tox_class"),
        "evaluation_mode": evaluation.get("evaluation_mode"),
    }


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": record["name"],
        "smiles": record["smiles"],
        "iteration": record["iteration"],
        "overall_score": record["overall_score"],
        "risk_score": record["risk_score"],
        "modification": record["modification"],
        "similarity_to_seed": record["similarity_to_seed"],
        "goal_status": record.get("goal_status"),
        "evaluation_mode": record["evaluation"].get("evaluation_mode"),
        "toxicity_score": record["evaluation"].get("toxicity", {}).get("toxicity_score"),
        "toxicity_class": record["evaluation"].get("toxicity", {}).get("predicted_tox_class"),
    }
