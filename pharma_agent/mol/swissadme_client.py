from __future__ import annotations

import json
import re
from threading import Lock
from time import monotonic, sleep
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag

from pharma_agent.config import SWISSADME_CACHE_PATH, settings


SWISSADME_URL = "https://www.swissadme.ch/index.php"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Origin": "https://www.swissadme.ch",
    "Referer": "https://www.swissadme.ch/",
}

_cache_lock = Lock()
_request_lock = Lock()
_last_request_time = 0.0


def query(smiles: str | list[str], timeout: int | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    single = isinstance(smiles, str)
    smiles_list = [smiles] if single else list(smiles)
    if not smiles_list:
        return {} if single else []
    if len(smiles_list) > 10:
        raise ValueError("SwissADME single request supports at most 10 SMILES.")

    timeout = timeout or settings.swissadme_timeout
    cache = _load_cache() if settings.swissadme_cache_enabled else {}

    results_by_smiles: dict[str, dict[str, Any]] = {}
    misses: list[str] = []
    for item in smiles_list:
        if item in cache:
            results_by_smiles[item] = cache[item]
        else:
            misses.append(item)

    if misses:
        html = _fetch(misses, timeout=timeout)
        parsed = _parse(html)
        if len(parsed) != len(misses):
            raise ValueError("SwissADME returned an unexpected number of results.")
        for smiles_item, parsed_item in zip(misses, parsed):
            results_by_smiles[smiles_item] = parsed_item
            if settings.swissadme_cache_enabled:
                cache[smiles_item] = parsed_item
        if settings.swissadme_cache_enabled:
            _save_cache(cache)

    ordered = [results_by_smiles[item] for item in smiles_list]
    return ordered[0] if single else ordered


def query_many(smiles_list: list[str], timeout: int | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index in range(0, len(smiles_list), 10):
        batch = smiles_list[index : index + 10]
        batch_result = query(batch, timeout=timeout)
        if isinstance(batch_result, dict):
            results.append(batch_result)
        else:
            results.extend(batch_result)
    return results


def _fetch(smiles_list: list[str], timeout: int) -> str:
    _throttle(settings.swissadme_delay_seconds)
    response = requests.post(
        SWISSADME_URL,
        headers=HEADERS,
        data={"smiles": "\n".join(smiles_list)},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def _parse(html: str) -> list[dict[str, Any]]:
    parsed = _try_parse_csv(html)
    if parsed:
        return parsed
    return _parse_tables(html)


def _try_parse_csv(html: str) -> list[dict[str, Any]]:
    header_match = re.search(
        r'textForClipBoard\s*=\s*textForClipBoard\s*\+\s*"(Molecule,Canonical SMILES,[^"]+)\\n"',
        html,
    )
    data_matches = re.findall(
        r'textForClipBoard\s*=\s*textForClipBoard\s*\+\s*"(Molecule \d+,[^"]+)\\n"',
        html,
    )
    if not header_match or not data_matches:
        return []

    headers = [item.strip() for item in header_match.group(1).split(",")]
    results: list[dict[str, Any]] = []
    for line in data_matches:
        values = [item.strip() for item in line.split(",")]
        row = {headers[idx]: values[idx] if idx < len(values) else None for idx in range(len(headers))}
        results.append(_build_from_csv(row))
    return results


def _build_from_csv(row: dict[str, Any]) -> dict[str, Any]:
    def get_value(key: str) -> str | None:
        value = row.get(key)
        return value if value else None

    return {
        "molecule_id": get_value("Molecule"),
        "physicochemical": {
            "canonical_smiles": get_value("Canonical SMILES"),
            "formula": get_value("Formula"),
            "molecular_weight": get_value("MW"),
            "num_heavy_atoms": get_value("#Heavy atoms"),
            "num_aromatic_heavy_atoms": get_value("#Aromatic heavy atoms"),
            "fraction_csp3": get_value("Fraction Csp3"),
            "num_rotatable_bonds": get_value("#Rotatable bonds"),
            "num_hbond_acceptors": get_value("#H-bond acceptors"),
            "num_hbond_donors": get_value("#H-bond donors"),
            "molar_refractivity": get_value("MR"),
            "tpsa": get_value("TPSA"),
        },
        "lipophilicity": {
            "ilogp": get_value("iLOGP"),
            "xlogp3": get_value("XLOGP3"),
            "wlogp": get_value("WLOGP"),
            "mlogp": get_value("MLOGP"),
            "silicos_it": get_value("Silicos-IT Log P"),
            "consensus": get_value("Consensus Log P"),
        },
        "water_solubility": {
            "esol_log_s": get_value("ESOL Log S"),
            "esol_mg_ml": get_value("ESOL Solubility (mg/ml)"),
            "esol_mol_l": get_value("ESOL Solubility (mol/l)"),
            "esol_class": get_value("ESOL Class"),
            "ali_log_s": get_value("Ali Log S"),
            "ali_mg_ml": get_value("Ali Solubility (mg/ml)"),
            "ali_mol_l": get_value("Ali Solubility (mol/l)"),
            "ali_class": get_value("Ali Class"),
            "silicos_it_log_sw": get_value("Silicos-IT LogSw"),
            "silicos_it_mg_ml": get_value("Silicos-IT Solubility (mg/ml)"),
            "silicos_it_mol_l": get_value("Silicos-IT Solubility (mol/l)"),
            "silicos_it_class": get_value("Silicos-IT class"),
        },
        "pharmacokinetics": {
            "gi_absorption": get_value("GI absorption"),
            "bbb_permeant": get_value("BBB permeant"),
            "pgp_substrate": get_value("Pgp substrate"),
            "cyp1a2": get_value("CYP1A2 inhibitor"),
            "cyp2c19": get_value("CYP2C19 inhibitor"),
            "cyp2c9": get_value("CYP2C9 inhibitor"),
            "cyp2d6": get_value("CYP2D6 inhibitor"),
            "cyp3a4": get_value("CYP3A4 inhibitor"),
            "log_kp_skin": get_value("log Kp (cm/s)"),
        },
        "druglikeness": {
            "lipinski_violations": get_value("Lipinski #violations"),
            "ghose_violations": get_value("Ghose #violations"),
            "veber_violations": get_value("Veber #violations"),
            "egan_violations": get_value("Egan #violations"),
            "muegge_violations": get_value("Muegge #violations"),
            "bioavailability_score": get_value("Bioavailability Score"),
        },
        "medicinal_chemistry": {
            "pains_alerts": get_value("PAINS #alerts"),
            "brenk_alerts": get_value("Brenk #alerts"),
            "leadlikeness_violations": get_value("Leadlikeness #violations"),
            "synthetic_accessibility": get_value("Synthetic Accessibility"),
        },
    }


def _parse_tables(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    results: list[dict[str, Any]] = []
    molecule_blocks = soup.find_all(
        "div",
        style=lambda value: value and "float: left; width: 940px" in value,
    )

    for index, block in enumerate(molecule_blocks, start=1):
        smiles_value = None
        for script_tag in block.find_all("script"):
            if script_tag.string:
                match = re.search(r'SMILES\["\d+"\]="([^"]+)"', script_tag.string)
                if match:
                    smiles_value = match.group(1)
                    break

        values: dict[str, str] = {}
        for row in block.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2 and not cells[0].get("bgcolor"):
                key = _cell_key(cells[0])
                value = re.sub(r"\s+", " ", cells[1].get_text()).strip().replace("\u00c5\u00b2", "Å²")
                if key and value:
                    values[key] = value

        logp_values = _extract_logp(block)
        get_value = lambda key: values.get(key)
        results.append(
            {
                "molecule_id": f"Molecule {index}",
                "physicochemical": {
                    "canonical_smiles": smiles_value,
                    "formula": get_value("Formula"),
                    "molecular_weight": get_value("Molecular weight"),
                    "num_heavy_atoms": get_value("Num. heavy atoms"),
                    "num_aromatic_heavy_atoms": get_value("Num. arom. heavy atoms"),
                    "fraction_csp3": get_value("Fraction Csp3"),
                    "num_rotatable_bonds": get_value("Num. rotatable bonds"),
                    "num_hbond_acceptors": get_value("Num. H-bond acceptors"),
                    "num_hbond_donors": get_value("Num. H-bond donors"),
                    "molar_refractivity": get_value("Molar Refractivity"),
                    "tpsa": get_value("TPSA"),
                },
                "lipophilicity": {
                    "ilogp": logp_values.get("ilogp"),
                    "xlogp3": logp_values.get("xlogp3"),
                    "wlogp": logp_values.get("wlogp"),
                    "mlogp": logp_values.get("mlogp"),
                    "silicos_it": logp_values.get("silicos_it"),
                    "consensus": get_value("Consensus Log Po/w"),
                },
                "water_solubility": {
                    "esol_log_s": get_value("Log S (ESOL)"),
                    "esol_class": _nth_value(block, "Class", 0),
                    "ali_log_s": get_value("Log S (Ali)"),
                    "ali_class": _nth_value(block, "Class", 1),
                    "silicos_it_log_sw": get_value("Log S (SILICOS-IT)"),
                    "silicos_it_class": _nth_value(block, "Class", 2),
                    "solubility": get_value("Solubility"),
                },
                "pharmacokinetics": {
                    "gi_absorption": get_value("GI absorption"),
                    "bbb_permeant": get_value("BBB permeant"),
                    "pgp_substrate": get_value("P-gp substrate"),
                    "cyp1a2": get_value("CYP1A2 inhibitor"),
                    "cyp2c19": get_value("CYP2C19 inhibitor"),
                    "cyp2c9": get_value("CYP2C9 inhibitor"),
                    "cyp2d6": get_value("CYP2D6 inhibitor"),
                    "cyp3a4": get_value("CYP3A4 inhibitor"),
                    "log_kp_skin": get_value("Log Kp (skin permeation)"),
                },
                "druglikeness": {
                    "lipinski_violations": get_value("Lipinski"),
                    "ghose_violations": get_value("Ghose"),
                    "veber_violations": get_value("Veber"),
                    "egan_violations": get_value("Egan"),
                    "muegge_violations": get_value("Muegge"),
                    "bioavailability_score": get_value("Bioavailability Score"),
                },
                "medicinal_chemistry": {
                    "pains_alerts": get_value("PAINS"),
                    "brenk_alerts": get_value("Brenk"),
                    "leadlikeness_violations": get_value("Leadlikeness"),
                    "synthetic_accessibility": get_value("Synthetic accessibility"),
                },
            }
        )
    return results


def _cell_key(cell: Tag) -> str:
    cloned = cell.__copy__()
    for link in cloned.find_all("a", class_="help"):
        link.decompose()
    return re.sub(r"\s+", " ", cloned.get_text()).strip()


def _extract_logp(block: Tag) -> dict[str, str | None]:
    keys = ["ilogp", "xlogp3", "wlogp", "mlogp", "silicos_it"]
    values: list[str] = []
    for row in block.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2 and _cell_key(cells[0]).startswith("Log"):
            value = re.sub(r"\s+", " ", cells[1].get_text()).strip()
            if value:
                values.append(value)
    return {keys[idx]: values[idx] if idx < len(values) else None for idx in range(len(keys))}


def _nth_value(block: Tag, row_text: str, occurrence: int) -> str | None:
    count = 0
    for row in block.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 2 and _cell_key(cells[0]) == row_text:
            if count == occurrence:
                return re.sub(r"\s+", " ", cells[1].get_text()).strip()
            count += 1
    return None


def _load_cache() -> dict[str, dict[str, Any]]:
    with _cache_lock:
        if not SWISSADME_CACHE_PATH.exists():
            return {}
        try:
            payload = json.loads(SWISSADME_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    with _cache_lock:
        SWISSADME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SWISSADME_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


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
