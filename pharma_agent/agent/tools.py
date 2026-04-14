from __future__ import annotations

from typing import Any
from xml.etree import ElementTree

import requests

from pharma_agent.config import settings
from pharma_agent.mol.evaluator import evaluate_smiles, evaluate_smiles_batch
from pharma_agent.rag.retriever import retrieve


PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str, retmax: int = 3) -> list[dict[str, Any]]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
        "tool": settings.pubmed_tool,
    }
    if settings.pubmed_email:
        params["email"] = settings.pubmed_email

    response = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=20)
    response.raise_for_status()
    ids = response.json().get("esearchresult", {}).get("idlist", [])

    results: list[dict[str, Any]] = []
    for pmid in ids:
        fetch_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "tool": settings.pubmed_tool,
        }
        if settings.pubmed_email:
            fetch_params["email"] = settings.pubmed_email

        item_response = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=fetch_params, timeout=20)
        item_response.raise_for_status()
        results.append(_parse_pubmed_xml(item_response.text, pmid))
    return results


def query_rules(question: str, top_k: int = 3) -> list[dict[str, Any]]:
    return retrieve(question, top_k=top_k)


def evaluate_molecule(smiles: str) -> dict[str, Any]:
    return evaluate_smiles(smiles)


def evaluate_molecule_batch(smiles_list: list[str], refine_top_n: int | None = None) -> list[dict[str, Any]]:
    return evaluate_smiles_batch(smiles_list, refine_top_n=refine_top_n)


def _parse_pubmed_xml(xml_text: str, fallback_pmid: str) -> dict[str, Any]:
    root = ElementTree.fromstring(xml_text)
    article = root.find(".//PubmedArticle")
    if article is None:
        return {
            "pmid": fallback_pmid,
            "title": "No title parsed",
            "abstract": "No abstract parsed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{fallback_pmid}/",
        }

    title = "".join(article.findtext(".//ArticleTitle", default="")).strip()
    abstract_parts = [elem.text.strip() for elem in article.findall(".//Abstract/AbstractText") if elem.text]
    journal = article.findtext(".//Journal/Title", default="").strip()
    year = article.findtext(".//PubDate/Year", default="").strip()

    return {
        "pmid": fallback_pmid,
        "title": title or "No title parsed",
        "journal": journal,
        "year": year,
        "abstract": " ".join(abstract_parts) if abstract_parts else "No abstract parsed",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{fallback_pmid}/",
    }
