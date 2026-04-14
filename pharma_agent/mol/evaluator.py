from __future__ import annotations

import math
import re
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from pharma_agent.config import settings
from pharma_agent.mol.protox3_client import predict as protox3_predict
from pharma_agent.mol.protox3_client import predict_many as protox3_predict_many
from pharma_agent.mol.swissadme_client import query as swissadme_query
from pharma_agent.mol.swissadme_client import query_many as swissadme_query_many


MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def evaluate_smiles(
    smiles: str,
    swissadme_result: dict[str, Any] | None = None,
    protox3_result: dict[str, Any] | None = None,
    *,
    include_protox: bool = True,
) -> dict[str, Any]:
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return {"valid": False, "error": "Invalid SMILES", "smiles": smiles}

    canonical_smiles = Chem.MolToSmiles(mol)
    evaluation_mode = "swissadme+rdkit+protox3" if include_protox else "swissadme+rdkit"
    warning_parts: list[str] = []

    if swissadme_result is None:
        try:
            swissadme_result = swissadme_query(canonical_smiles)
        except Exception as exc:
            swissadme_result = None
            evaluation_mode = "rdkit-fallback+protox3" if include_protox else "rdkit-fallback"
            warning_parts.append(f"SwissADME unavailable: {exc}")

    if include_protox and protox3_result is None:
        try:
            protox3_result = protox3_predict(canonical_smiles)
        except Exception as exc:
            protox3_result = None
            warning_parts.append(f"ProTox3 unavailable: {exc}")
            if evaluation_mode == "swissadme+rdkit+protox3":
                evaluation_mode = "swissadme+rdkit"
            elif evaluation_mode == "rdkit-fallback+protox3":
                evaluation_mode = "rdkit-fallback"

    mw_rdkit = round(Descriptors.MolWt(mol), 2)
    logp_rdkit = round(Descriptors.MolLogP(mol), 2)
    hbd_rdkit = Lipinski.NumHDonors(mol)
    hba_rdkit = Lipinski.NumHAcceptors(mol)
    tpsa_rdkit = round(rdMolDescriptors.CalcTPSA(mol), 2)
    rotb_rdkit = Lipinski.NumRotatableBonds(mol)
    fraction_csp3 = round(rdMolDescriptors.CalcFractionCSP3(mol), 2)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    qed = round(QED.qed(mol), 3)
    bertz = round(Descriptors.BertzCT(mol), 2)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

    physicochemical = (swissadme_result or {}).get("physicochemical", {})
    lipophilicity = (swissadme_result or {}).get("lipophilicity", {})
    pk = (swissadme_result or {}).get("pharmacokinetics", {})
    druglikeness = (swissadme_result or {}).get("druglikeness", {})
    medchem = (swissadme_result or {}).get("medicinal_chemistry", {})

    mw = _first_float(physicochemical.get("molecular_weight"), mw_rdkit)
    consensus_logp = _first_float(lipophilicity.get("consensus"), logp_rdkit)
    tpsa = _first_float(physicochemical.get("tpsa"), tpsa_rdkit)
    rotb = _first_int(physicochemical.get("num_rotatable_bonds"), rotb_rdkit)
    hbd = _first_int(physicochemical.get("num_hbond_donors"), hbd_rdkit)
    hba = _first_int(physicochemical.get("num_hbond_acceptors"), hba_rdkit)
    sa = _first_float(medchem.get("synthetic_accessibility"), _sa_proxy_from_bertz(bertz))
    bioavailability = _first_float(druglikeness.get("bioavailability_score"), 0.45)

    lipinski_violations = _violation_count(
        druglikeness.get("lipinski_violations"),
        int(mw > 500) + int(consensus_logp > 5) + int(hbd > 5) + int(hba > 10),
    )
    veber_violations = _violation_count(
        druglikeness.get("veber_violations"),
        int(tpsa > 140 or rotb > 10),
    )
    ghose_violations = _violation_count(druglikeness.get("ghose_violations"), 0)
    egan_violations = _violation_count(druglikeness.get("egan_violations"), 0)
    muegge_violations = _violation_count(druglikeness.get("muegge_violations"), 0)
    leadlikeness_violations = _violation_count(medchem.get("leadlikeness_violations"), 0)
    pains_alerts = _violation_count(medchem.get("pains_alerts"), 0)
    brenk_alerts = _violation_count(medchem.get("brenk_alerts"), 0)

    gi_absorption = _normalize_flag(pk.get("gi_absorption"))
    bbb_permeant = _normalize_flag(pk.get("bbb_permeant"))
    pgp_substrate = _normalize_flag(pk.get("pgp_substrate"))
    cyp_flags = {
        "cyp1a2": _normalize_flag(pk.get("cyp1a2")),
        "cyp2c19": _normalize_flag(pk.get("cyp2c19")),
        "cyp2c9": _normalize_flag(pk.get("cyp2c9")),
        "cyp2d6": _normalize_flag(pk.get("cyp2d6")),
        "cyp3a4": _normalize_flag(pk.get("cyp3a4")),
    }
    cyp_inhibitor_count = sum(1 for value in cyp_flags.values() if value == "yes")

    metric_details = {
        "molecular_weight": _lower_better_metric(mw, 500.0, "MW"),
        "consensus_logp": _lower_better_metric(consensus_logp, 5.0, "Consensus LogP"),
        "tpsa": _lower_better_metric(tpsa, 140.0, "TPSA"),
        "rotatable_bonds": _lower_better_metric(float(rotb), 10.0, "RotB"),
        "synthetic_accessibility": _lower_better_metric(sa, 6.0, "SA"),
        "bioavailability": _higher_better_metric(bioavailability, 0.55, "Bioavailability"),
    }

    balance_score = round(
        (
            metric_details["molecular_weight"]["score"]
            + metric_details["consensus_logp"]["score"]
            + metric_details["tpsa"]["score"]
            + metric_details["rotatable_bonds"]["score"]
        )
        / 4,
        1,
    )
    absorption_score = round(
        (
            (100.0 if gi_absorption == "high" else 52.0)
            + metric_details["bioavailability"]["score"]
            + (12.0 if pgp_substrate != "yes" else 0.0)
        )
        / 2.12,
        1,
    )
    druglikeness_score = round(
        _clamp(
            100.0
            - lipinski_violations * 18.0
            - veber_violations * 14.0
            - ghose_violations * 6.0
            - egan_violations * 6.0
            - muegge_violations * 6.0
            - leadlikeness_violations * 5.0,
            5.0,
            100.0,
        ),
        1,
    )
    medchem_score = round(
        _clamp(
            100.0
            - pains_alerts * 22.0
            - brenk_alerts * 10.0
            - cyp_inhibitor_count * 4.0
            - (8.0 if pgp_substrate == "yes" else 0.0),
            5.0,
            100.0,
        ),
        1,
    )
    base_risk_score = round(
        _clamp(
            8.0
            + lipinski_violations * 15.0
            + veber_violations * 10.0
            + pains_alerts * 22.0
            + brenk_alerts * 10.0
            + cyp_inhibitor_count * 5.0
            + (12.0 if gi_absorption != "high" else 0.0)
            + max(0.0, consensus_logp - 4.0) * 8.0
            + max(0.0, sa - 6.0) * 10.0
            + (8.0 if pgp_substrate == "yes" else 0.0),
            5.0,
            95.0,
        ),
        1,
    )
    base_overall_score = round(
        _clamp(
            balance_score * 0.30
            + absorption_score * 0.16
            + druglikeness_score * 0.20
            + medchem_score * 0.14
            + metric_details["synthetic_accessibility"]["score"] * 0.10
            + metric_details["bioavailability"]["score"] * 0.04
            + qed * 100.0 * 0.03
            + (100.0 - base_risk_score) * 0.03,
            18.0,
            97.0,
        ),
        1,
    )

    toxicity_bundle = _build_toxicity_bundle(protox3_result)
    if toxicity_bundle["included"]:
        metric_details["toxicity_profile"] = _higher_better_metric(
            toxicity_bundle["toxicity_score"],
            65.0,
            "ProTox3",
        )
        risk_score = round(
            _clamp(base_risk_score * 0.76 + toxicity_bundle["toxicity_risk_score"] * 0.24, 5.0, 95.0),
            1,
        )
        overall_score = round(
            _clamp(base_overall_score * 0.82 + toxicity_bundle["toxicity_score"] * 0.18, 18.0, 97.0),
            1,
        )
    else:
        risk_score = base_risk_score
        overall_score = base_overall_score

    overall_grade = _grade(overall_score)
    strengths, risks = _build_narrative(
        gi_absorption=gi_absorption,
        lipinski_violations=lipinski_violations,
        veber_violations=veber_violations,
        pains_alerts=pains_alerts,
        brenk_alerts=brenk_alerts,
        pgp_substrate=pgp_substrate,
        consensus_logp=consensus_logp,
        sa=sa,
        bioavailability=bioavailability,
        toxicity_bundle=toxicity_bundle,
    )

    swiss_compact = {
        "gi_absorption": gi_absorption,
        "bbb_permeant": bbb_permeant,
        "pgp_substrate": pgp_substrate,
        "bioavailability_score": bioavailability,
        "lipinski_violations": lipinski_violations,
        "veber_violations": veber_violations,
        "pains_alerts": pains_alerts,
        "brenk_alerts": brenk_alerts,
        "leadlikeness_violations": leadlikeness_violations,
        "synthetic_accessibility": sa,
        "cyp_inhibitors": cyp_flags,
        "cyp_inhibitor_count": cyp_inhibitor_count,
    }

    evaluation_warning = " | ".join(part for part in warning_parts if part) or None
    return {
        "valid": True,
        "smiles": canonical_smiles,
        "input_smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "core_scaffold_smiles": scaffold or "",
        "molecular_weight": mw,
        "logp": consensus_logp,
        "rdkit_logp": logp_rdkit,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rotatable_bonds": rotb,
        "heavy_atoms": heavy_atoms,
        "aromatic_rings": aromatic_rings,
        "fraction_csp3": fraction_csp3,
        "qed": qed,
        "bertz_complexity": bertz,
        "sa_proxy": sa,
        "lipinski_pass": lipinski_violations == 0,
        "veber_pass": veber_violations == 0,
        "lipinski_violations": lipinski_violations,
        "veber_violations": veber_violations,
        "ghose_violations": ghose_violations,
        "egan_violations": egan_violations,
        "muegge_violations": muegge_violations,
        "leadlikeness_violations": leadlikeness_violations,
        "pains_alerts": pains_alerts,
        "brenk_alerts": brenk_alerts,
        "gi_absorption": gi_absorption,
        "bbb_permeant": bbb_permeant,
        "pgp_substrate": pgp_substrate,
        "bioavailability_score": bioavailability,
        "cyp_inhibitors": cyp_flags,
        "cyp_inhibitor_count": cyp_inhibitor_count,
        "balance_score": balance_score,
        "absorption_score": absorption_score,
        "druglikeness_score": druglikeness_score,
        "medchem_score": medchem_score,
        "base_risk_score": base_risk_score,
        "base_overall_score": base_overall_score,
        "risk_score": risk_score,
        "overall_score": overall_score,
        "overall_grade": overall_grade,
        "metric_details": metric_details,
        "strengths": strengths,
        "risks": risks,
        "evaluation_mode": evaluation_mode,
        "evaluation_warning": evaluation_warning,
        "swissadme": swiss_compact,
        "toxicity": toxicity_bundle,
    }


def evaluate_smiles_batch(smiles_list: list[str], refine_top_n: int | None = None) -> list[dict[str, Any]]:
    canonical_by_input: dict[str, str] = {}
    valid_canonical: list[str] = []
    invalid_results: dict[str, dict[str, Any]] = {}

    for smiles in smiles_list:
        mol = _mol_from_smiles(smiles)
        if mol is None:
            invalid_results[smiles] = {"valid": False, "error": "Invalid SMILES", "smiles": smiles}
            continue
        canonical = Chem.MolToSmiles(mol)
        canonical_by_input[smiles] = canonical
        if canonical not in valid_canonical:
            valid_canonical.append(canonical)

    swiss_map: dict[str, dict[str, Any]] = {}
    swiss_error: str | None = None
    if valid_canonical:
        try:
            swiss_results = swissadme_query_many(valid_canonical)
            swiss_map = {smiles: result for smiles, result in zip(valid_canonical, swiss_results)}
        except Exception as exc:
            swiss_error = str(exc)

    results: list[dict[str, Any]] = []
    for smiles in smiles_list:
        if smiles in invalid_results:
            results.append(invalid_results[smiles])
            continue
        canonical = canonical_by_input[smiles]
        item = evaluate_smiles(
            canonical,
            swissadme_result=swiss_map.get(canonical),
            include_protox=False,
        )
        if swiss_error and item.get("evaluation_warning") is None:
            item["evaluation_warning"] = f"SwissADME unavailable: {swiss_error}"
        results.append(item)

    valid_indices = [idx for idx, item in enumerate(results) if item.get("valid")]
    if not valid_indices:
        return results

    refine_count = max(0, refine_top_n if refine_top_n is not None else settings.protox3_refine_top_n)
    refine_count = min(refine_count, len(valid_indices))
    if refine_count == 0:
        return results

    ranked_indices = sorted(
        valid_indices,
        key=lambda idx: (
            -float(results[idx].get("base_overall_score", results[idx].get("overall_score", 0.0))),
            float(results[idx].get("base_risk_score", results[idx].get("risk_score", 100.0))),
            idx,
        ),
    )
    selected_indices = ranked_indices[:refine_count]
    selected_smiles = [results[idx]["canonical_smiles"] for idx in selected_indices]
    protox_results = protox3_predict_many(selected_smiles)

    for idx, protox_result in zip(selected_indices, protox_results):
        canonical = results[idx]["canonical_smiles"]
        if protox_result.get("valid"):
            refined = evaluate_smiles(
                canonical,
                swissadme_result=swiss_map.get(canonical),
                protox3_result=protox_result,
                include_protox=True,
            )
        else:
            refined = evaluate_smiles(
                canonical,
                swissadme_result=swiss_map.get(canonical),
                include_protox=False,
            )
            refined["evaluation_warning"] = _combine_warnings(
                refined.get("evaluation_warning"),
                f"ProTox3 unavailable: {protox_result.get('error', 'unknown error')}",
            )
        results[idx] = refined

    if selected_indices and not any(results[idx].get("toxicity", {}).get("included") for idx in selected_indices):
        rescue_idx = ranked_indices[0]
        rescue_smiles = results[rescue_idx]["canonical_smiles"]
        try:
            rescue_protox = protox3_predict(rescue_smiles)
            results[rescue_idx] = evaluate_smiles(
                rescue_smiles,
                swissadme_result=swiss_map.get(rescue_smiles),
                protox3_result=rescue_protox,
                include_protox=True,
            )
        except Exception as exc:
            results[rescue_idx]["evaluation_warning"] = _combine_warnings(
                results[rescue_idx].get("evaluation_warning"),
                f"ProTox3 rescue failed: {exc}",
            )

    return results


def preserves_core(seed_smiles: str, candidate_smiles: str) -> bool:
    seed_mol = _mol_from_smiles(seed_smiles)
    candidate_mol = _mol_from_smiles(candidate_smiles)
    if seed_mol is None or candidate_mol is None:
        return False
    seed_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=seed_mol)
    candidate_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=candidate_mol)
    if seed_scaffold and candidate_scaffold:
        return seed_scaffold == candidate_scaffold
    return scaffold_similarity(seed_smiles, candidate_smiles) >= 0.55


def scaffold_similarity(smiles_a: str, smiles_b: str) -> float:
    mol_a = _mol_from_smiles(smiles_a)
    mol_b = _mol_from_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = MORGAN_GENERATOR.GetFingerprint(mol_a)
    fp_b = MORGAN_GENERATOR.GetFingerprint(mol_b)
    return round(float(DataStructs.TanimotoSimilarity(fp_a, fp_b)), 4)


def _build_toxicity_bundle(protox3_result: dict[str, Any] | None) -> dict[str, Any]:
    if not protox3_result or not protox3_result.get("valid"):
        return {
            "included": False,
            "toxicity_score": None,
            "toxicity_risk_score": None,
            "predicted_ld50_mg_kg": None,
            "predicted_tox_class": None,
            "pred_accuracy_pct": None,
            "avg_similarity_pct": None,
            "active_model_count": None,
            "high_confidence_active_count": None,
            "organ_toxicity_active_count": None,
            "toxicity_endpoint_active_count": None,
            "cyp_active_count": None,
            "medium_or_high_target_count": None,
            "high_target_count": None,
            "headline": "ProTox3 unavailable",
            "summary_lines": ["No ProTox3 toxicity refinement was applied to this molecule."],
            "raw": None,
        }

    summary = protox3_result.get("summary", {})
    tox_class = _to_int(summary.get("predicted_tox_class"))
    ld50 = _to_float(summary.get("predicted_ld50_mg_kg"))
    pred_accuracy = _to_float(summary.get("pred_accuracy_pct")) or 0.0
    similarity_pct = _to_float(summary.get("avg_similarity_pct")) or 0.0
    active_models = _to_int(summary.get("active_model_count")) or 0
    high_conf_active = _to_int(summary.get("high_confidence_active_count")) or 0
    organ_active = _to_int(summary.get("organ_toxicity_active_count")) or 0
    endpoint_active = _to_int(summary.get("toxicity_endpoint_active_count")) or 0
    cyp_active = _to_int(summary.get("cyp_active_count")) or 0
    medium_targets = _to_int(summary.get("medium_or_high_target_count")) or 0
    high_targets = _to_int(summary.get("high_target_count")) or 0

    class_component = 50.0
    if tox_class is not None:
        class_component = 20.0 + (tox_class - 1) * 16.0

    ld50_component = 50.0
    if ld50 is not None and ld50 > 0:
        ld50_component = _clamp(math.log10(ld50 + 1.0) / math.log10(5001.0) * 100.0, 5.0, 100.0)

    toxicity_score = _clamp(
        0.46 * class_component
        + 0.24 * ld50_component
        + 0.10 * pred_accuracy
        + 0.06 * similarity_pct
        + 12.0
        - active_models * 1.3
        - high_conf_active * 2.6
        - organ_active * 4.8
        - endpoint_active * 2.8
        - cyp_active * 1.4
        - medium_targets * 3.0
        - high_targets * 4.5,
        5.0,
        96.0,
    )
    toxicity_risk_score = _clamp(100.0 - toxicity_score, 5.0, 95.0)

    summary_lines: list[str] = []
    if tox_class is not None:
        summary_lines.append(f"Predicted toxicity class {tox_class} (higher is safer in ProTox3).")
    if ld50 is not None:
        summary_lines.append(f"Predicted LD50 {ld50:.0f} mg/kg.")
    if organ_active > 0:
        summary_lines.append(f"{organ_active} organ toxicity model(s) flagged active.")
    if endpoint_active > 0:
        summary_lines.append(f"{endpoint_active} toxicity endpoint model(s) flagged active.")
    if medium_targets > 0:
        summary_lines.append(f"{medium_targets} toxicity target hit(s) reached medium or high binding.")
    if not summary_lines:
        summary_lines.append("No major ProTox3 toxicity signal was highlighted.")

    headline = "Integrated ProTox3 toxicity profile"
    if tox_class is not None and tox_class <= 3:
        headline = "ProTox3 flagged elevated toxicity risk"
    elif tox_class is not None and tox_class >= 5 and organ_active == 0 and high_targets == 0:
        headline = "ProTox3 profile looks comparatively clean"

    return {
        "included": True,
        "toxicity_score": round(float(toxicity_score), 1),
        "toxicity_risk_score": round(float(toxicity_risk_score), 1),
        "predicted_ld50_mg_kg": ld50,
        "predicted_tox_class": tox_class,
        "pred_accuracy_pct": round(float(pred_accuracy), 1) if pred_accuracy else None,
        "avg_similarity_pct": round(float(similarity_pct), 1) if similarity_pct else None,
        "active_model_count": active_models,
        "high_confidence_active_count": high_conf_active,
        "organ_toxicity_active_count": organ_active,
        "toxicity_endpoint_active_count": endpoint_active,
        "cyp_active_count": cyp_active,
        "medium_or_high_target_count": medium_targets,
        "high_target_count": high_targets,
        "headline": headline,
        "summary_lines": summary_lines[:4],
        "raw": protox3_result,
    }


def _lower_better_metric(value: float, limit: float, label: str) -> dict[str, Any]:
    ratio = value / limit if limit else 0.0
    if value <= limit * 0.8:
        status = "pass"
        color = "#1D9E75"
    elif value <= limit:
        status = "warning"
        color = "#EF9F27"
    else:
        status = "fail"
        color = "#C83C36"
    if value <= limit:
        score = 100.0 - max(0.0, ratio - 0.5) * 55.0
    else:
        score = 72.0 - ((value - limit) / max(limit, 1.0)) * 120.0
    return {
        "label": label,
        "value": round(float(value), 2),
        "limit": round(float(limit), 2),
        "ratio": round(min(max(ratio, 0.0), 1.5), 4),
        "score": round(_clamp(score, 5.0, 100.0), 1),
        "status": status,
        "color": color,
    }


def _higher_better_metric(value: float, target: float, label: str) -> dict[str, Any]:
    ratio = value / target if target else 0.0
    if value >= target:
        status = "pass"
        color = "#1D9E75"
    elif value >= target * 0.75:
        status = "warning"
        color = "#EF9F27"
    else:
        status = "fail"
        color = "#C83C36"
    return {
        "label": label,
        "value": round(float(value), 2),
        "limit": round(float(target), 2),
        "ratio": round(min(max(ratio, 0.0), 1.2), 4),
        "score": round(_clamp(100.0 * min(max(ratio, 0.0), 1.0), 5.0, 100.0), 1),
        "status": status,
        "color": color,
    }


def _build_narrative(
    *,
    gi_absorption: str,
    lipinski_violations: int,
    veber_violations: int,
    pains_alerts: int,
    brenk_alerts: int,
    pgp_substrate: str,
    consensus_logp: float,
    sa: float,
    bioavailability: float,
    toxicity_bundle: dict[str, Any],
) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    risks: list[str] = []

    if lipinski_violations == 0:
        strengths.append("SwissADME shows no Lipinski violation, so oral developability is acceptable at baseline.")
    if veber_violations == 0:
        strengths.append("Veber filters are satisfied, which helps exposure and permeability balance.")
    if gi_absorption == "high":
        strengths.append("GI absorption is predicted as high.")
    if bioavailability >= 0.55:
        strengths.append("Bioavailability score reaches the common medicinal chemistry cut-off.")

    if pains_alerts > 0:
        risks.append("PAINS alerts indicate possible screening interference risk.")
    if brenk_alerts > 0:
        risks.append("Brenk alerts suggest potentially problematic substructures.")
    if pgp_substrate == "yes":
        risks.append("P-gp substrate liability may reduce intracellular exposure.")
    if consensus_logp > 4.5:
        risks.append("Consensus LogP is high and may hurt solubility or safety.")
    if sa > 6.0:
        risks.append("Synthetic accessibility is weak and may complicate route design.")
    if lipinski_violations > 0 or veber_violations > 0:
        risks.append("Drug-likeness rule violations remain and should be repaired in follow-up design.")

    if toxicity_bundle["included"]:
        tox_class = toxicity_bundle.get("predicted_tox_class")
        ld50 = toxicity_bundle.get("predicted_ld50_mg_kg")
        organ_count = toxicity_bundle.get("organ_toxicity_active_count") or 0
        high_targets = toxicity_bundle.get("high_target_count") or 0
        high_conf = toxicity_bundle.get("high_confidence_active_count") or 0

        if tox_class is not None and tox_class >= 5:
            strengths.append(f"ProTox3 predicts toxicity class {tox_class}, which is comparatively favorable.")
        if ld50 is not None and ld50 >= 2000:
            strengths.append("Predicted LD50 is relatively high, indicating lower acute toxicity pressure.")
        if organ_count == 0 and high_targets == 0:
            strengths.append("No prominent organ toxicity or high-binding toxicity target signal was observed in ProTox3.")

        if tox_class is not None and tox_class <= 3:
            risks.append(f"ProTox3 predicts toxicity class {tox_class}, which is a material safety warning.")
        if ld50 is not None and ld50 < 300:
            risks.append("Predicted LD50 is low and indicates elevated acute toxicity risk.")
        if organ_count > 0:
            risks.append(f"ProTox3 flagged {organ_count} active organ toxicity model(s).")
        if high_conf > 3:
            risks.append(f"Multiple high-confidence toxicity models are active ({high_conf}).")
        if high_targets > 0:
            risks.append(f"{high_targets} toxicity target(s) reached high binding in ProTox3.")

    if not strengths:
        strengths.append("No decisive medicinal chemistry strength is obvious yet.")
    if not risks:
        risks.append("No major medicinal chemistry red flag is obvious, but wet-lab confirmation is still required.")
    return strengths[:5], risks[:5]


def _grade(score: float) -> str:
    if score >= 88:
        return "A"
    if score >= 80:
        return "B+"
    if score >= 72:
        return "B"
    if score >= 64:
        return "C+"
    if score >= 56:
        return "C"
    return "D"


def _sa_proxy_from_bertz(bertz: float) -> float:
    return round(min(10.0, max(1.0, 1.0 + bertz / 210.0)), 2)


def _first_float(value: Any, default: float) -> float:
    if value is None:
        return round(float(default), 2)
    if isinstance(value, (float, int)):
        return round(float(value), 2)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return round(float(match.group(0)), 2) if match else round(float(default), 2)


def _first_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, int):
        return value
    match = re.search(r"-?\d+", str(value))
    return int(match.group(0)) if match else int(default)


def _violation_count(value: Any, default: int) -> int:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"no", "none", "0"}:
        return 0
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return default


def _normalize_flag(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in {"high", "low"}:
        return text
    if text in {"yes", "y", "true"}:
        return "yes"
    if text in {"no", "n", "false"}:
        return "no"
    return text or "unknown"


def _combine_warnings(existing: str | None, extra: str | None) -> str | None:
    if existing and extra:
        return f"{existing} | {extra}"
    return existing or extra


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"-?\d+", str(value))
    return int(match.group(0)) if match else None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
