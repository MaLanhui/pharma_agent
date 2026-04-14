from __future__ import annotations

import base64
import json
import textwrap
from io import BytesIO
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pharma_agent.agent.core import continue_manual_optimization, run_auto_optimization, start_manual_optimization, summarize_session
from pharma_agent.config import INDEX_DIR, settings


st.set_page_config(page_title="靶点化合物优化 Agent", layout="wide")


def html_block(html: str) -> None:
    st.html(textwrap.dedent(html).strip())


def inject_styles() -> None:
    html_block(
        """
        <style>
        :root{
          --bg:#f5f7fb;--surface:#ffffff;--surface-soft:#f7f9fc;--surface-strong:#eef3f9;--border:#d8e0ea;
          --text:#17212b;--muted:#67778a;--blue:#2563eb;--green:#15966b;--amber:#d27a14;--red:#cb3b33;
          --radius:20px;--radius-sm:14px;--input-bg:#f7f9fc;--input-text:#17212b;
        }
        .stApp{
          background:
            radial-gradient(circle at top left,rgba(37,99,235,.10),transparent 26%),
            radial-gradient(circle at top right,rgba(21,150,107,.10),transparent 22%),
            linear-gradient(180deg,#fcfdff 0%,var(--bg) 100%);
        }
        section.main > div.block-container{max-width:1360px;padding-top:5.2rem;padding-bottom:2rem;}
        .page-header{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;margin-bottom:18px;}
        .page-kicker{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--blue);font-weight:700;margin-bottom:8px;}
        .page-title{margin:0;font-size:34px;line-height:1.08;color:var(--text);font-weight:800;}
        .page-subtitle{margin-top:10px;color:var(--muted);font-size:14px;line-height:1.7;max-width:840px;}
        .github-link{display:inline-flex;align-items:center;gap:10px;text-decoration:none;color:var(--text);background:rgba(255,255,255,.86);border:1px solid var(--border);padding:10px 14px;border-radius:999px;white-space:nowrap;}
        .github-link svg{width:18px;height:18px;fill:currentColor;}
        .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1rem 1.15rem;margin-bottom:14px;}
        .card-soft{background:linear-gradient(180deg,rgba(255,255,255,.97) 0%,rgba(247,249,252,.95) 100%);}
        .section-label{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:8px;font-weight:700;}
        .title{font-size:18px;font-weight:800;color:var(--text);margin-bottom:8px;}
        .body{font-size:13px;line-height:1.7;color:var(--muted);}
        .mono{font-family:Consolas,monospace;font-size:11px;color:var(--muted);word-break:break-all;}
        .badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;margin-right:6px;margin-bottom:6px;}
        .b-info{background:#eaf2ff;color:var(--blue)} .b-ok{background:#eaf8f1;color:var(--green)} .b-warn{background:#fff4e7;color:var(--amber)} .b-bad{background:#fdecec;color:var(--red)}
        .mol-box{height:270px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface-soft);display:flex;align-items:center;justify-content:center;overflow:hidden;margin:12px 0;}
        .mol-box img{max-width:100%;max-height:100%;display:block;}
        .metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px;}
        .metric{background:var(--surface-soft);border-radius:var(--radius-sm);padding:.8rem .95rem;}
        .metric-label{font-size:12px;color:var(--muted);margin-bottom:4px;}
        .metric-value{font-size:20px;font-weight:800;color:var(--text);}
        .metric-value.pass{color:var(--green)} .metric-value.warn{color:var(--amber)} .metric-value.fail{color:var(--red)}
        .bar{margin-bottom:10px;}
        .bar-head{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-bottom:4px;}
        .bar-track{height:8px;background:#e7edf5;border-radius:999px;overflow:hidden;}
        .bar-fill{height:100%;border-radius:999px;}
        .suggestion{border:1px solid var(--border);border-radius:var(--radius-sm);padding:.85rem .95rem;margin-bottom:8px;background:rgba(255,255,255,.72);}
        .suggestion-title{font-size:13px;font-weight:800;color:var(--text);margin-bottom:5px;}
        .suggestion-body{font-size:13px;line-height:1.65;color:var(--muted);}
        .delta{font-size:13px;color:var(--muted);margin-top:8px;}
        .goal-row{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid rgba(216,224,234,.65);}
        .goal-row:last-child{border-bottom:none;padding-bottom:0;}
        .goal-name{font-size:13px;color:var(--text);font-weight:600;}
        .goal-value{font-size:12px;color:var(--muted);text-align:right;}
        .goal-pass{color:var(--green);font-weight:700;}
        .goal-fail{color:var(--red);font-weight:700;}
        [data-testid="stForm"]{background:rgba(255,255,255,.78);border:1px solid var(--border);border-radius:var(--radius);padding:1rem 1rem .4rem 1rem;margin-bottom:16px;}
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input{
          background:var(--input-bg)!important;color:var(--input-text)!important;border:1px solid var(--border)!important;border-radius:12px!important;
        }
        [data-testid="stRadio"] div[role="radiogroup"]{
          background:var(--surface-soft);border:1px solid var(--border);border-radius:12px;padding:4px 8px;
        }
        [data-testid="stRadio"] label{padding:6px 4px!important;}
        [data-testid="stSlider"] [data-baseweb="slider"]{padding-left:8px;padding-right:8px;}
        [data-testid="stMarkdownContainer"] p, label, [data-testid="stWidgetLabel"]{color:var(--text)!important;}
        @media (max-width:960px){
          .page-header{flex-direction:column}
          .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
        }
        </style>
        """
    )


def ensure_state() -> None:
    defaults: dict[str, Any] = {
        "analysis_result": None,
        "manual_session": None,
        "manual_signature": None,
        "analysis_phase": "idle",
        "pending_request": None,
        "run_error": None,
        "candidate_display_limit": 8,
        "form_target": "",
        "form_mode_label": "自动迭代",
        "form_lead_smiles": "",
        "form_goal_score": 84,
        "form_max_risk_score": 42,
        "form_min_toxicity_score": 58,
        "form_min_toxicity_class": 4,
        "form_min_bioavailability_score": 0.45,
        "form_max_iterations": 6,
        "form_candidate_count": 6,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    html_block(
        """
        <div class="page-header">
          <div>
            <div class="page-kicker">Medicinal Chemistry Workflow</div>
            <h1 class="page-title">靶点化合物优化 Agent</h1>
            <div class="page-subtitle">面向靶点 lead 的自动/手动结构优化、ADME 预筛和 ProTox3 毒性精排工作台。</div>
          </div>
          <a class="github-link" href="https://github.com/Malanhui" target="_blank" rel="noopener noreferrer">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.5 7.5 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"></path></svg>
            <span>github.com/Malanhui</span>
          </a>
        </div>
        """
    )


def fmt(value: Any, digits: int = 1, fallback: str = "-") -> str:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if digits else f"{int(value)}"
    return str(value)


def badge_class(value: str) -> str:
    if value in {"A", "B+", "B"}:
        return "b-ok"
    if value in {"C+", "C"}:
        return "b-warn"
    return "b-bad"


def metric_class(status: str) -> str:
    return {"pass": "pass", "warning": "warn", "fail": "fail"}.get(status, "")


def image_as_base64(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(520, 320))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def phase_button_label() -> str:
    return {
        "idle": "启动 Agent 分析",
        "running": "分析进行中...",
        "completed": "开始新的分析",
        "error": "重新尝试分析",
    }.get(st.session_state.analysis_phase, "启动 Agent 分析")


def queue_request(payload: dict[str, Any]) -> None:
    st.session_state.pending_request = payload
    st.session_state.analysis_phase = "running"
    st.session_state.run_error = None
    st.session_state.analysis_result = None
    st.session_state.manual_session = None
    st.session_state.manual_signature = None
    st.rerun()


def execute_pending_request() -> None:
    payload = st.session_state.pending_request
    if not payload:
        return
    with st.spinner("正在检索证据、生成候选并执行 SwissADME / ProTox3 评分..."):
        try:
            common_kwargs = {
                "goal_score": payload["goal_score"],
                "max_risk_score": payload["max_risk_score"],
                "min_toxicity_score": payload["min_toxicity_score"],
                "min_toxicity_class": payload["min_toxicity_class"],
                "min_bioavailability_score": payload["min_bioavailability_score"],
                "max_iterations": payload["max_iterations"],
                "candidate_count": payload["candidate_count"],
            }
            if payload["mode_label"] == "自动迭代":
                result = run_auto_optimization(payload["target"], payload["lead_smiles"], **common_kwargs)
                st.session_state.analysis_result = result
            else:
                result = start_manual_optimization(payload["target"], payload["lead_smiles"], **common_kwargs)
                st.session_state.analysis_result = result
                st.session_state.manual_session = result["session"]
                st.session_state.manual_signature = payload["signature"]
            st.session_state.analysis_phase = "completed"
            st.session_state.run_error = None
        except Exception as exc:
            st.session_state.analysis_phase = "error"
            st.session_state.run_error = str(exc)
        finally:
            st.session_state.pending_request = None
    st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("运行状态")
        st.write(f"本地索引: {'已构建' if (INDEX_DIR / 'index.faiss').exists() else '未构建'}")
        st.write(f"Embedding 模式: `{settings.local_embedding_mode}`")
        st.write(f"候选生成 API: {'已配置' if settings.deepseek_api_key else '未配置'}")
        st.write("评分主链: `SwissADME + RDKit + ProTox3`")
        st.write(f"SwissADME 节流: `{settings.swissadme_delay_seconds:.1f}s/次`")
        st.write(f"ProTox3 节流: `{settings.protox3_delay_seconds:.1f}s/次`")
        st.write(f"ProTox3 重试: `{settings.protox3_max_retries}` 次")
        phase_map = {"idle": "待命", "running": "运行中", "completed": "已完成", "error": "运行失败"}
        st.write(f"任务状态: `{phase_map.get(st.session_state.analysis_phase, st.session_state.analysis_phase)}`")


def render_form() -> None:
    running = st.session_state.analysis_phase == "running"
    with st.form("optimization_form", clear_on_submit=False):
        col1, col2 = st.columns([1.2, 1.1], gap="medium")
        with col1:
            target = st.text_input("靶点名称", key="form_target", placeholder="EGFR / BRAF / CDK4", disabled=running)
        with col2:
            mode_options = ["自动迭代", "手动逐轮"]
            mode_index = mode_options.index(st.session_state.form_mode_label) if st.session_state.form_mode_label in mode_options else 0
            mode_label = st.radio("优化模式", mode_options, horizontal=True, index=mode_index, key="form_mode_label", disabled=running)

        lead_smiles = st.text_area("起始 lead 分子 SMILES", key="form_lead_smiles", height=110, placeholder="输入一个已知针对该靶点的 lead 分子 SMILES", disabled=running)

        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            goal_score = st.slider("目标综合分", min_value=60, max_value=96, key="form_goal_score", step=1, disabled=running)
        with c2:
            max_risk_score = st.slider("目标风险分上限", min_value=10, max_value=90, key="form_max_risk_score", step=1, disabled=running)
        with c3:
            min_toxicity_score = st.slider("目标 ProTox3 分", min_value=0, max_value=95, key="form_min_toxicity_score", step=1, disabled=running)

        c4, c5, c6 = st.columns(3, gap="medium")
        with c4:
            min_toxicity_class = st.slider("目标毒性等级下限", min_value=0, max_value=6, key="form_min_toxicity_class", step=1, disabled=running, help="0 表示不启用该约束。")
        with c5:
            min_bioavailability_score = st.slider("目标 Bioavailability", min_value=0.0, max_value=1.0, key="form_min_bioavailability_score", step=0.05, disabled=running)
        with c6:
            candidate_display_limit = st.number_input("候选池展示上限", min_value=2, max_value=16, value=int(st.session_state.candidate_display_limit), step=1, disabled=running)

        c7, c8 = st.columns(2, gap="medium")
        with c7:
            max_iterations = st.number_input("最大迭代轮数", min_value=1, max_value=12, key="form_max_iterations", step=1, disabled=running)
        with c8:
            candidate_count = st.number_input("每轮候选数", min_value=2, max_value=10, key="form_candidate_count", step=1, disabled=running)

        submitted = st.form_submit_button(phase_button_label(), type="primary", disabled=running, width="stretch")

    if submitted:
        if not target.strip():
            st.session_state.analysis_phase = "error"
            st.session_state.run_error = "请输入靶点名称。"
        elif not lead_smiles.strip():
            st.session_state.analysis_phase = "error"
            st.session_state.run_error = "请输入起始 lead 分子 SMILES。"
        elif not (INDEX_DIR / "index.faiss").exists():
            st.session_state.analysis_phase = "error"
            st.session_state.run_error = "本地 FAISS 索引不存在，请先运行 `python -m pharma_agent.rag.build_index`。"
        else:
            st.session_state.candidate_display_limit = int(candidate_display_limit)
            queue_request(
                {
                    "target": target.strip(),
                    "lead_smiles": lead_smiles.strip(),
                    "mode_label": mode_label,
                    "goal_score": float(goal_score),
                    "max_risk_score": float(max_risk_score),
                    "min_toxicity_score": float(min_toxicity_score) if float(min_toxicity_score) > 0 else None,
                    "min_toxicity_class": int(min_toxicity_class) if int(min_toxicity_class) > 0 else None,
                    "min_bioavailability_score": float(min_bioavailability_score) if float(min_bioavailability_score) > 0 else None,
                    "max_iterations": int(max_iterations),
                    "candidate_count": int(candidate_count),
                    "signature": (
                        target.strip(),
                        lead_smiles.strip(),
                        mode_label,
                        int(goal_score),
                        int(max_iterations),
                        int(candidate_count),
                    ),
                }
            )

    if st.session_state.run_error:
        st.error(st.session_state.run_error)
    elif running:
        st.info("当前任务正在运行，旧结果已清空。外部服务已启用节流和重试，耗时会比之前更长。")


def molecule_card(record: dict[str, Any], *, section: str, subtitle: str = "", delta: str = "") -> str:
    evaluation = record["evaluation"]
    toxicity = evaluation.get("toxicity", {})
    img64 = image_as_base64(record["smiles"])
    image_html = f'<img src="data:image/png;base64,{img64}" alt="{section}" />' if img64 else '<div class="body">无法生成结构图</div>'
    badges = [
        f'<span class="badge {badge_class(evaluation["overall_grade"])}">Grade {evaluation["overall_grade"]}</span>',
        f'<span class="badge b-info">Score {evaluation["overall_score"]}</span>',
        f'<span class="badge b-warn">Risk {evaluation["risk_score"]}</span>',
    ]
    if toxicity.get("included"):
        badges.append(f'<span class="badge b-ok">ProTox3 {toxicity.get("toxicity_score")}</span>')
        if toxicity.get("predicted_tox_class") is not None:
            badges.append(f'<span class="badge b-info">Class {toxicity.get("predicted_tox_class")}</span>')
    if evaluation.get("evaluation_warning"):
        badges.append('<span class="badge b-warn">部分外部服务回退</span>')
    subtitle_html = f'<div class="body">{subtitle}</div>' if subtitle else ""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return f"""
    <div class="card card-soft">
      <div class="section-label">{section}</div>
      <div class="title">{record["name"]}</div>
      {subtitle_html}
      <div>{"".join(badges)}</div>
      <div class="mol-box">{image_html}</div>
      <div class="body">轮次：第 {record["iteration"]} 轮</div>
      <div class="body">改造方向：{record["modification"]}</div>
      <div class="body">设计理由：{record["rationale"]}</div>
      {delta_html}
      <div class="mono">SMILES: {record["smiles"]}</div>
    </div>
    """


def score_card(result: dict[str, Any]) -> str:
    ev = result["best_record"]["evaluation"]
    tox = ev.get("toxicity", {})
    metrics = [
        ("综合评分", str(ev["overall_score"]), "pass" if ev["overall_grade"] in {"A", "B+", "B"} else "warn"),
        ("基础评分", str(ev.get("base_overall_score", "-")), "pass"),
        ("GI Absorption", str(ev.get("gi_absorption", "unknown")).upper(), "pass" if ev.get("gi_absorption") == "high" else "warn"),
        ("Bioavailability", fmt(ev.get("bioavailability_score"), 2), metric_class(ev["metric_details"]["bioavailability"]["status"])),
        ("Lipinski", "通过" if ev["lipinski_pass"] else "未通过", "pass" if ev["lipinski_pass"] else "fail"),
        ("ProTox3", fmt(tox.get("toxicity_score"), 1), "pass" if (tox.get("toxicity_score") or 0) >= 65 else "warn"),
        ("毒性等级", str(tox.get("predicted_tox_class", "-")), "pass" if (tox.get("predicted_tox_class") or 0) >= 5 else "warn"),
        ("LD50", f"{fmt(tox.get('predicted_ld50_mg_kg'), 0)} mg/kg", "pass" if (tox.get("predicted_ld50_mg_kg") or 0) >= 2000 else "warn"),
    ]
    metric_html = "".join(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value {css}">{value}</div></div>' for label, value, css in metrics)
    bars = [
        ("分子量", ev["metric_details"]["molecular_weight"]),
        ("Consensus LogP", ev["metric_details"]["consensus_logp"]),
        ("TPSA", ev["metric_details"]["tpsa"]),
        ("合成可及性", ev["metric_details"]["synthetic_accessibility"]),
        ("Bioavailability", ev["metric_details"]["bioavailability"]),
    ]
    if "toxicity_profile" in ev["metric_details"]:
        bars.append(("ProTox3 毒性档案", ev["metric_details"]["toxicity_profile"]))
    bar_html = "".join(
        f'<div class="bar"><div class="bar-head"><span>{label}</span><span>{meta["value"]} / {meta["limit"]}</span></div><div class="bar-track"><div class="bar-fill" style="width:{min(meta["ratio"] * 100, 100):.0f}%;background:{meta["color"]}"></div></div></div>'
        for label, meta in bars
    )
    strengths = "；".join(ev.get("strengths", []))
    risks = "；".join(ev.get("risks", []))
    return f"""
    <div class="card">
      <div class="section-label">综合评分卡</div>
      <div class="metric-grid">{metric_html}</div>
      {bar_html}
      <div class="body"><strong>优点：</strong>{strengths}</div>
      <div class="body"><strong>风险：</strong>{risks}</div>
      <div class="body"><strong>停止原因：</strong>{result['stop_reason'] or '未停止'}</div>
    </div>
    """


def toxicity_card(result: dict[str, Any]) -> str:
    tox = result["best_record"]["evaluation"].get("toxicity", {})
    if not tox.get("included"):
        return """
        <div class="card">
          <div class="section-label">ProTox3 毒性精排</div>
          <div class="title">当前结果未拿到 ProTox3 明细</div>
          <div class="body">客户端已经启用缓存、节流和重试，但本轮仍可能因为服务风控、超时或临时不可用而回退到非毒性精排结果。</div>
        </div>
        """
    metrics = [
        ("Toxicity Score", fmt(tox.get("toxicity_score"), 1), "pass" if (tox.get("toxicity_score") or 0) >= 65 else "warn"),
        ("Predicted Class", str(tox.get("predicted_tox_class", "-")), "pass" if (tox.get("predicted_tox_class") or 0) >= 5 else "warn"),
        ("LD50", f"{fmt(tox.get('predicted_ld50_mg_kg'), 0)} mg/kg", "pass" if (tox.get("predicted_ld50_mg_kg") or 0) >= 2000 else "warn"),
        ("Prediction Accuracy", f"{fmt(tox.get('pred_accuracy_pct'), 1)}%", "pass"),
        ("Active Models", str(tox.get("active_model_count", "-")), "warn"),
        ("Organ Toxicity", str(tox.get("organ_toxicity_active_count", "-")), "warn"),
        ("High-Conf Actives", str(tox.get("high_confidence_active_count", "-")), "warn"),
        ("Toxicity Targets", str(tox.get("medium_or_high_target_count", "-")), "warn"),
    ]
    metric_html = "".join(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value {css}">{value}</div></div>' for label, value, css in metrics)
    summary_html = "".join(f"<div class='body'>• {line}</div>" for line in tox.get("summary_lines", []))
    return f"""
    <div class="card">
      <div class="section-label">ProTox3 毒性精排</div>
      <div class="title">{tox.get("headline", "ProTox3 summary")}</div>
      <div class="metric-grid">{metric_html}</div>
      {summary_html}
    </div>
    """


def goal_card(result: dict[str, Any]) -> str:
    goal_status = result.get("goal_status") or {}
    rows = []
    for item in goal_status.get("items", []):
        if not item.get("enabled"):
            continue
        comparator = "≥" if item["mode"] == "min" else "≤"
        css = "goal-pass" if item["passed"] else "goal-fail"
        rows.append(
            f"""
            <div class="goal-row">
              <div class="goal-name">{item['label']}</div>
              <div class="goal-value">
                当前 {fmt(item['actual'], 2)} / 目标 {comparator} {fmt(item['target'], 2)}<br />
                <span class="{css}">{'已达成' if item['passed'] else '未达成'}</span>
              </div>
            </div>
            """
        )
    summary = f"{goal_status.get('met_count', 0)} / {goal_status.get('total_count', 0)} 项目标已达成"
    return f"""
    <div class="card">
      <div class="section-label">目标约束</div>
      <div class="title">{summary}</div>
      <div>{''.join(rows)}</div>
    </div>
    """


def report_card(report: dict[str, Any]) -> str:
    suggestions = []
    for idx, item in enumerate(report.get("suggestions", []), start=1):
        refs = " ".join(f'<span class="badge b-info">引用{n}</span>' for n in item.get("citation_numbers", []))
        suggestions.append(f'<div class="suggestion"><div class="suggestion-title">[{idx}] {item["headline"]}</div><div class="suggestion-body">{item["recommendation"]}</div><div style="margin-top:8px">{refs}</div></div>')
    return f"""
    <div class="card">
      <div class="section-label">优化结论</div>
      <div class="title">综合评级：{report['overall_grade']}</div>
      <div class="body"><strong>靶点摘要：</strong>{report['target_summary']}</div>
      <div class="body"><strong>迭代摘要：</strong>{report['optimization_summary']}</div>
      <div class="body"><strong>分子结论：</strong>{report['molecule_assessment']}</div>
      <div class="body" style="margin-top:8px"><strong>总评：</strong>{report['overall_comment']}</div>
      <div style="margin-top:10px">{''.join(suggestions)}</div>
    </div>
    """


def history_chart(history: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    labels = [x["label"] for x in history]
    fig.add_trace(go.Scatter(x=labels, y=[x["current_score"] for x in history], mode="lines+markers", name="本轮最佳", line={"color": "#2563EB", "width": 3}, marker={"size": 8}))
    fig.add_trace(go.Scatter(x=labels, y=[x["best_score"] for x in history], mode="lines+markers", name="全局最佳", line={"color": "#15966B", "width": 3}, marker={"size": 8}))
    fig.add_trace(go.Scatter(x=labels, y=[x["risk_score"] for x in history], mode="lines+markers", name="风险分", line={"color": "#D27A14", "width": 3}, marker={"size": 8}))
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 12, "b": 10}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0})
    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,120,120,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,120,120,0.12)", range=[0, 100])
    return fig


def candidate_table(result: dict[str, Any], limit: int) -> pd.DataFrame:
    rows = []
    for item in result.get("latest_candidates", [])[:limit]:
        ev = item["evaluation"]
        tox = ev.get("toxicity", {})
        goal_status = item.get("goal_status", {})
        rows.append(
            {
                "候选": item["name"],
                "SMILES": item["smiles"],
                "综合分": item["overall_score"],
                "基础分": ev.get("base_overall_score"),
                "风险分": item["risk_score"],
                "Bioavailability": ev.get("bioavailability_score"),
                "ProTox3": tox.get("toxicity_score"),
                "毒性等级": tox.get("predicted_tox_class"),
                "LD50(mg/kg)": tox.get("predicted_ld50_mg_kg"),
                "Goal Fit": f"{goal_status.get('met_count', 0)}/{goal_status.get('total_count', 0)}",
                "Similarity": item.get("similarity_to_seed"),
                "改造方向": item.get("modification"),
            }
        )
    return pd.DataFrame(rows)


def trace_table(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "步骤": item["step"],
            "工具": item["tool"],
            "摘要": item["summary"],
            "状态": item.get("status"),
            "耗时(ms)": item.get("duration_ms"),
            "输入": json.dumps(item.get("tool_input"), ensure_ascii=False),
            "输出": json.dumps(item.get("tool_output"), ensure_ascii=False),
        }
        for item in result.get("steps", [])
    ])


def citations_table(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "引用": paper["citation_number"],
            "PMID": paper["pmid"],
            "标题": paper["title"],
            "期刊": paper.get("journal"),
            "年份": paper.get("year"),
            "链接": paper.get("url"),
        }
        for paper in result.get("report", {}).get("paper_citations", [])
    ])


def toxicity_models_table(result: dict[str, Any]) -> pd.DataFrame:
    raw = result["best_record"]["evaluation"].get("toxicity", {}).get("raw") or {}
    return pd.DataFrame(raw.get("toxicity_models", []))


def toxicity_targets_table(result: dict[str, Any]) -> pd.DataFrame:
    raw = result["best_record"]["evaluation"].get("toxicity", {}).get("raw") or {}
    return pd.DataFrame([
        {"靶点": key, "Binding Score": info.get("binding_score"), "Binding Label": info.get("binding_label"), "UniProt": info.get("uniprot")}
        for key, info in (raw.get("toxicity_targets") or {}).items()
    ])


def render_results(result: dict[str, Any]) -> None:
    seed = result["seed_record"]
    best = result["best_record"]
    score_delta = best["overall_score"] - seed["overall_score"]
    risk_delta = best["risk_score"] - seed["risk_score"]

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html_block(molecule_card(seed, section="起始分子", subtitle=f"靶点：{result['target']}"))
    with c2:
        html_block(molecule_card(best, section="最优分子", subtitle=f"当前最优候选，来源于第 {best['iteration']} 轮", delta=f"相对起始分子：综合分 {score_delta:+.1f}，风险分 {risk_delta:+.1f}"))

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        html_block(score_card(result))
    with right:
        html_block(toxicity_card(result))
        html_block(goal_card(result))

    html_block(report_card(result["report"]))

    st.subheader("迭代历史")
    st.plotly_chart(history_chart(result["history"]), width="stretch", config={"displayModeBar": False})

    limit = int(st.session_state.candidate_display_limit or 8)
    df = candidate_table(result, limit)
    if not df.empty:
        st.subheader("最新一轮候选池")
        st.dataframe(df, width="stretch", hide_index=True)

    if result["mode"] == "manual" and st.session_state.manual_session and not result["stopped"] and result.get("latest_candidates"):
        st.subheader("手动继续迭代")
        options = {f"{item['name']} | Score {item['overall_score']} | {item['modification']}": item["smiles"] for item in result["latest_candidates"][:limit]}
        selected = st.selectbox("选择下一轮种子分子", list(options.keys()))
        m1, m2 = st.columns(2, gap="medium")
        with m1:
            if st.button("使用所选候选继续下一轮", width="stretch", type="primary"):
                with st.spinner("正在继续手动迭代..."):
                    try:
                        updated = continue_manual_optimization(st.session_state.manual_session, options[selected])
                        st.session_state.analysis_result = updated
                        st.session_state.manual_session = updated["session"]
                        st.session_state.analysis_phase = "completed"
                    except Exception as exc:
                        st.session_state.analysis_phase = "error"
                        st.session_state.run_error = f"继续迭代失败: {exc}"
                    st.rerun()
        with m2:
            if st.button("停止手动迭代并保留当前最优", width="stretch"):
                st.session_state.manual_session["stopped"] = True
                st.session_state.manual_session["stop_reason"] = "用户手动停止，并保留当前最优分子。"
                st.session_state.analysis_result = summarize_session(st.session_state.manual_session)
                st.session_state.analysis_phase = "completed"
                st.rerun()

    cite_df = citations_table(result)
    if not cite_df.empty:
        with st.expander("查看文献引用"):
            st.dataframe(cite_df, width="stretch", hide_index=True)
    tox_targets_df = toxicity_targets_table(result)
    if not tox_targets_df.empty:
        with st.expander("查看 ProTox3 毒性靶点"):
            st.dataframe(tox_targets_df, width="stretch", hide_index=True)
    tox_models_df = toxicity_models_table(result)
    if not tox_models_df.empty:
        with st.expander("查看 ProTox3 模型明细"):
            st.dataframe(tox_models_df, width="stretch", hide_index=True)
    with st.expander("查看执行链路"):
        st.dataframe(trace_table(result), width="stretch", hide_index=True)
    with st.expander("查看本地规则片段"):
        for rule in result["report"].get("rule_citations", []):
            st.markdown(f"**{rule['evidence_id']}**  score=`{rule.get('score', '-')}`")
            st.write(rule["text"])


inject_styles()
ensure_state()
render_header()
render_sidebar()
render_form()
execute_pending_request()

if st.session_state.analysis_result:
    render_results(st.session_state.analysis_result)
else:
    st.info("输入靶点和起始 lead 分子 SMILES 后启动分析。自动模式会持续迭代到达到目标约束或达到轮数上限；手动模式每轮只展示候选池，由你决定下一轮种子。")
