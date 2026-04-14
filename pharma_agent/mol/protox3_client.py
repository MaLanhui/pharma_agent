from __future__ import annotations

import json
import re
from threading import Lock
from time import monotonic, sleep
from typing import Any

import requests
from bs4 import BeautifulSoup
from rdkit import Chem
from rdkit.Chem import AllChem

from pharma_agent.config import PROTOX3_CACHE_PATH, settings


BASE_URL = "https://tox.charite.de/protox3"
STEP1_URL = f"{BASE_URL}/index.php?site=compound_search_similarity"
STEP2_URL = f"{BASE_URL}/src/run_models.php"

ALL_MODELS = [
    "dili",
    "neuro",
    "nephro",
    "respi",
    "cardio",
    "carcino",
    "immuno",
    "mutagen",
    "cyto",
    "bbb",
    "eco",
    "clinical",
    "nutri",
    "nr_ahr",
    "nr_ar",
    "nr_ar_lbd",
    "nr_aromatase",
    "nr_er",
    "nr_er_lbd",
    "nr_ppar_gamma",
    "sr_are",
    "sr_hse",
    "sr_mmp",
    "sr_p53",
    "sr_atad5",
    "mie_thr_alpha",
    "mie_thr_beta",
    "mie_ttr",
    "mie_ryr",
    "mie_gabar",
    "mie_nmdar",
    "mie_ampar",
    "mie_kar",
    "mie_ache",
    "mie_car",
    "mie_pxr",
    "mie_nadhox",
    "mie_vgsc",
    "mie_nis",
    "CYP1A2",
    "CYP2C19",
    "CYP2C9",
    "CYP2D6",
    "CYP3A4",
    "CYP2E1",
]

MODEL_META = {
    "dili": ("Organ Toxicity", "Hepatotoxicity (DILI)"),
    "neuro": ("Organ Toxicity", "Neurotoxicity"),
    "nephro": ("Organ Toxicity", "Nephrotoxicity"),
    "respi": ("Organ Toxicity", "Respiratory toxicity"),
    "cardio": ("Organ Toxicity", "Cardiotoxicity"),
    "carcino": ("Toxicity Endpoints", "Carcinogenicity"),
    "immuno": ("Toxicity Endpoints", "Immunotoxicity"),
    "mutagen": ("Toxicity Endpoints", "Mutagenicity"),
    "cyto": ("Toxicity Endpoints", "Cytotoxicity"),
    "bbb": ("Toxicity Endpoints", "BBB Permeability"),
    "eco": ("Toxicity Endpoints", "Ecotoxicity"),
    "clinical": ("Toxicity Endpoints", "Clinical Toxicity"),
    "nutri": ("Toxicity Endpoints", "Nutritional Toxicity"),
    "nr_ahr": ("Tox21 Nuclear Receptor", "Aryl hydrocarbon receptor"),
    "nr_ar": ("Tox21 Nuclear Receptor", "Androgen receptor"),
    "nr_ar_lbd": ("Tox21 Nuclear Receptor", "Androgen receptor LBD"),
    "nr_aromatase": ("Tox21 Nuclear Receptor", "Aromatase"),
    "nr_er": ("Tox21 Nuclear Receptor", "Estrogen receptor alpha"),
    "nr_er_lbd": ("Tox21 Nuclear Receptor", "Estrogen receptor LBD"),
    "nr_ppar_gamma": ("Tox21 Nuclear Receptor", "PPAR-gamma"),
    "sr_are": ("Tox21 Stress Response", "Nrf2/ARE"),
    "sr_hse": ("Tox21 Stress Response", "Heat shock factor"),
    "sr_mmp": ("Tox21 Stress Response", "Mitochondrial membrane potential"),
    "sr_p53": ("Tox21 Stress Response", "Tumor suppressor p53"),
    "sr_atad5": ("Tox21 Stress Response", "ATAD5"),
    "mie_thr_alpha": ("Mol. Initiating Events", "Thyroid hormone receptor alpha"),
    "mie_thr_beta": ("Mol. Initiating Events", "Thyroid hormone receptor beta"),
    "mie_ttr": ("Mol. Initiating Events", "Transthyretin"),
    "mie_ryr": ("Mol. Initiating Events", "Ryanodine receptor"),
    "mie_gabar": ("Mol. Initiating Events", "GABA receptor"),
    "mie_nmdar": ("Mol. Initiating Events", "NMDA receptor"),
    "mie_ampar": ("Mol. Initiating Events", "AMPA receptor"),
    "mie_kar": ("Mol. Initiating Events", "Kainate receptor"),
    "mie_ache": ("Mol. Initiating Events", "Acetylcholinesterase"),
    "mie_car": ("Mol. Initiating Events", "Constitutive androstane receptor"),
    "mie_pxr": ("Mol. Initiating Events", "Pregnane X receptor"),
    "mie_nadhox": ("Mol. Initiating Events", "NADH-quinone oxidoreductase"),
    "mie_vgsc": ("Mol. Initiating Events", "Voltage-gated sodium channel"),
    "mie_nis": ("Mol. Initiating Events", "Na+/I- symporter"),
    "CYP1A2": ("CYP Metabolism", "Cytochrome CYP1A2"),
    "CYP2C19": ("CYP Metabolism", "Cytochrome CYP2C19"),
    "CYP2C9": ("CYP Metabolism", "Cytochrome CYP2C9"),
    "CYP2D6": ("CYP Metabolism", "Cytochrome CYP2D6"),
    "CYP3A4": ("CYP Metabolism", "Cytochrome CYP3A4"),
    "CYP2E1": ("CYP Metabolism", "Cytochrome CYP2E1"),
}

TARGET_COLOR_MAP = {
    "tg-698h": (0, "No binding"),
    "tg-zfch": (1, "Low binding"),
    "tg-sxfm": (2, "Medium binding"),
    "tg-lg9c": (3, "High binding"),
    "tg-cxkv": (0, "Not tested"),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "Referer": f"{BASE_URL}/index.php?site=compound_input",
    "Origin": BASE_URL,
}

_cache_lock = Lock()
_request_lock = Lock()
_last_request_time = 0.0


def predict(
    smiles: str,
    timeout: int | None = None,
    verbose: bool = False,
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES for ProTox3 prediction.")

    canonical_smiles = Chem.MolToSmiles(mol)
    cache = _load_cache() if settings.protox3_cache_enabled else {}
    if canonical_smiles in cache:
        return cache[canonical_smiles]

    timeout = timeout or settings.protox3_timeout
    mol_block = _smiles_to_mol_block(canonical_smiles)
    owns_session = session is None
    request_session = session or requests.Session()
    request_session.headers.update(HEADERS)
    try:
        if verbose:
            print("[ProTox3] step1 start")
        response1 = _post_with_retries(
            request_session,
            STEP1_URL,
            {"smilesString": mol_block, "defaultName": "", "smiles": canonical_smiles, "pubchem_name": ""},
            timeout,
            request_name="step1",
            verbose=verbose,
        )
        parsed = _parse_html(response1.text)
        parsed["smiles"] = canonical_smiles

        server_id = parsed.get("server_id")
        if not server_id:
            raise RuntimeError("ProTox3 did not return server_id.")

        if verbose:
            print("[ProTox3] step2 start")
        response2 = _post_with_retries(
            request_session,
            STEP2_URL,
            {"models": " ".join(ALL_MODELS), "sdfile": "empty", "mol": mol_block, "id": server_id},
            timeout,
            request_name="step2",
            verbose=verbose,
        )
        try:
            models_json = json.loads(response2.text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"ProTox3 returned invalid JSON: {response2.text[:300]}") from exc

        parsed["toxicity_models"] = _parse_models(models_json)
        parsed["summary"] = _build_summary(parsed)
        parsed["valid"] = True

        if settings.protox3_cache_enabled:
            cache[canonical_smiles] = parsed
            _save_cache(cache)
        return parsed
    finally:
        if owns_session:
            request_session.close()


def predict_many(smiles_list: list[str], timeout: int | None = None, verbose: bool = False) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    request_session = requests.Session()
    request_session.headers.update(HEADERS)
    for smiles in smiles_list:
        try:
            results.append(predict(smiles, timeout=timeout, verbose=verbose, session=request_session))
        except Exception as exc:
            results.append({"valid": False, "smiles": smiles, "error": str(exc)})
    request_session.close()
    return results


def _smiles_to_mol_block(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    AllChem.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol)


def _parse_html(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    parsed: dict[str, Any] = {}

    server_match = re.search(r"var\s+server_id\s*=\s*'([^']+)'", html)
    parsed["server_id"] = server_match.group(1) if server_match else None

    def find_h1(pattern: str) -> str | None:
        for tag in soup.find_all("h1"):
            match = re.search(pattern, tag.get_text())
            if match:
                return match.group(1)
        return None

    parsed["predicted_ld50_mg_kg"] = _to_float(find_h1(r"Predicted LD50:\s*([\d.]+)"))
    parsed["predicted_tox_class"] = _to_int(find_h1(r"Predicted Toxicity Class:\s*(\d+)"))
    parsed["avg_similarity_pct"] = _to_float(find_h1(r"Average similarity:\s*([\d.]+)"))
    parsed["pred_accuracy_pct"] = _to_float(find_h1(r"Prediction accuracy:\s*([\d.]+)"))
    parsed["molecular_properties"] = _parse_properties_table(soup)
    parsed["similar_compounds"] = _parse_similar_compounds(soup)
    parsed["toxicity_targets"] = _parse_toxicity_targets(soup)
    return parsed


def _parse_properties_table(soup: BeautifulSoup) -> dict[str, str]:
    property_map = {
        "Molweight": "molweight",
        "Number of hydrogen bond acceptors": "hba",
        "Number of hydrogen bond donors": "hbd",
        "Number of atoms": "num_atoms",
        "Number of bonds": "num_bonds",
        "Number of rotable bonds": "rotatable_bonds",
        "Molecular refractivity": "mr",
        "Topological Polar Surface Area": "tpsa",
        "octanol/water partition coefficient(logP)": "logp",
    }
    properties: dict[str, str] = {}
    for table in soup.find_all("table", id="table-out"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            key = cells[0].get_text(strip=True)
            if key in property_map:
                properties[property_map[key]] = cells[1].get_text(strip=True)
        if properties:
            break
    return properties


def _parse_similar_compounds(soup: BeautifulSoup) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    tables = soup.find_all("table", id="table-out")
    for table in tables[1:]:
        compound: dict[str, str] = {}
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            key = cells[0].get_text(strip=True)
            if key in {"Formula", "molweight", "endpoint", "tox class, avg", "tox class, min"}:
                compound[key] = cells[1].get_text(strip=True)
        if compound:
            records.append(compound)
    return records


def _parse_toxicity_targets(soup: BeautifulSoup) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    table = soup.find("table", class_="tg")
    if table is None:
        return targets

    rows = table.find_all("tr")
    if len(rows) < 2:
        return targets
    headers = rows[0].find_all("th")
    values = rows[1].find_all("td")
    for header, cell in zip(headers, values):
        target_name = header.get_text(strip=True)
        link = header.find("a")
        css_class = (cell.get("class") or ["tg-cxkv"])[0]
        binding_score, binding_label = TARGET_COLOR_MAP.get(css_class, (0, "Unknown"))
        targets[target_name] = {
            "uniprot": link["href"] if link and link.has_attr("href") else "",
            "binding_score": binding_score,
            "binding_label": binding_label,
        }
    return targets


def _parse_models(json_data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model_id in ALL_MODELS:
        category, model_name = MODEL_META.get(model_id, ("Other", model_id))
        if model_id in json_data:
            raw = json_data[model_id]
            prediction = "Active" if str(raw.get("Prediction", "")) in {"1", "1.0"} else "Inactive"
            probability = _to_float(raw.get("Probability"))
            confidence = "High" if probability is not None and probability >= 0.7 else "Low"
        else:
            prediction = "Below threshold"
            probability = None
            confidence = "N/A"
        records.append(
            {
                "model_id": model_id,
                "category": category,
                "model_name": model_name,
                "prediction": prediction,
                "probability": round(probability, 4) if probability is not None else None,
                "confidence": confidence,
            }
        )
    return records


def _build_summary(result: dict[str, Any]) -> dict[str, Any]:
    models = result.get("toxicity_models", [])
    active_models = [item for item in models if item.get("prediction") == "Active"]
    high_confidence_active = [
        item for item in active_models if item.get("probability") is not None and item["probability"] >= 0.7
    ]
    organ_toxicity_active = [item for item in active_models if item.get("category") == "Organ Toxicity"]
    endpoint_active = [item for item in active_models if item.get("category") == "Toxicity Endpoints"]
    cyp_active = [item for item in active_models if item.get("category") == "CYP Metabolism"]
    target_hits = list(result.get("toxicity_targets", {}).values())
    medium_or_high_targets = [item for item in target_hits if item.get("binding_score", 0) >= 2]
    high_targets = [item for item in target_hits if item.get("binding_score", 0) >= 3]

    return {
        "active_model_count": len(active_models),
        "high_confidence_active_count": len(high_confidence_active),
        "organ_toxicity_active_count": len(organ_toxicity_active),
        "toxicity_endpoint_active_count": len(endpoint_active),
        "cyp_active_count": len(cyp_active),
        "medium_or_high_target_count": len(medium_or_high_targets),
        "high_target_count": len(high_targets),
        "predicted_ld50_mg_kg": result.get("predicted_ld50_mg_kg"),
        "predicted_tox_class": result.get("predicted_tox_class"),
        "pred_accuracy_pct": result.get("pred_accuracy_pct"),
        "avg_similarity_pct": result.get("avg_similarity_pct"),
    }


def _throttle(delay_seconds: float) -> None:
    global _last_request_time
    if delay_seconds <= 0:
        return
    with _request_lock:
        now = monotonic()
        wait_seconds = delay_seconds - (now - _last_request_time)
        if wait_seconds > 0:
            sleep(wait_seconds)
        _last_request_time = monotonic()


def _post_with_retries(
    session: requests.Session,
    url: str,
    data: dict[str, Any],
    timeout: int,
    *,
    request_name: str,
    verbose: bool,
) -> requests.Response:
    last_error: Exception | None = None
    attempts = max(1, settings.protox3_max_retries)
    for attempt in range(1, attempts + 1):
        try:
            _throttle(settings.protox3_delay_seconds)
            response = session.post(url, data=data, timeout=timeout)
            response.raise_for_status()
            _validate_response_body(response.text, request_name=request_name)
            return response
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            backoff = settings.protox3_retry_backoff_seconds * attempt
            if verbose:
                print(f"[ProTox3] {request_name} attempt {attempt} failed: {exc}; retry in {backoff:.1f}s")
            sleep(backoff)
    raise RuntimeError(f"ProTox3 {request_name} failed after {attempts} attempts: {last_error}") from last_error


def _validate_response_body(body: str, *, request_name: str) -> None:
    stripped = body.strip()
    if not stripped:
        raise RuntimeError(f"{request_name} returned empty response")
    lower = stripped.lower()
    risk_markers = ["too many requests", "access denied", "temporarily unavailable", "cloudflare", "<title>error"]
    if any(marker in lower for marker in risk_markers):
        raise RuntimeError(f"{request_name} appears rate-limited or blocked")


def _load_cache() -> dict[str, dict[str, Any]]:
    with _cache_lock:
        if not PROTOX3_CACHE_PATH.exists():
            return {}
        try:
            payload = json.loads(PROTOX3_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    with _cache_lock:
        PROTOX3_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROTOX3_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


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
