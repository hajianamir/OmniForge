"""Multi-provider AI Copilot Agent supporting OpenAI, Anthropic, Azure, Ollama, and offline deterministic planning."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from omniforge.copilot.memory import CopilotMemory
from omniforge.copilot.prompts import SYSTEM_PROMPT, TOOL_CALLING_INSTRUCTION
from omniforge.copilot.tools import CopilotToolRegistry
from omniforge.core.logging import get_logger

logger = get_logger(__name__)


class CopilotResponse(BaseModel):
    """Structured multi-turn response emitted by :class:`OmniForgeCopilotAgent`."""

    reply: str
    provider_used: str
    tools_executed: list[dict[str, Any]] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)


class OmniForgeCopilotAgent:
    """Enterprise AI Copilot controlling all OmniForge workflows via English natural language.

    Supports live LLM providers (``openai``, ``anthropic``, ``azure``, ``ollama``) and a
    zero-dependency ``local`` deterministic intent-and-tool planner that executes real OmniForge
    analytics end-to-end offline.
    """

    def __init__(
        self,
        *,
        provider: str = "local",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        ollama_base_url: str = "http://localhost:11434",
        azure_endpoint: str | None = None,
        memory: CopilotMemory | None = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.ollama_base_url = ollama_base_url
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.memory = memory or CopilotMemory()
        self.tools = CopilotToolRegistry(self.memory)

    def load_dataframe(self, df: pd.DataFrame, name: str = "active_dataset") -> None:
        """Attach an in-memory DataFrame to the Copilot's working context."""
        self.memory.set_dataset(name, df)

    def chat(self, user_message: str) -> CopilotResponse:
        """Process an English natural-language instruction, invoke OmniForge tools, and return findings."""
        self.memory.add_turn("user", user_message)

        if self.provider != "local" and self._provider_credentials_available():
            try:
                response = self._chat_with_llm(user_message)
                self.memory.add_turn("assistant", response.reply)
                return response
            except Exception as exc:
                logger.warning("copilot_llm_fallback_to_local", provider=self.provider, reason=str(exc))

        response = self._chat_local_planner(user_message)
        self.memory.add_turn("assistant", response.reply)
        return response

    def _provider_credentials_available(self) -> bool:
        if self.provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        if self.provider == "anthropic":
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        if self.provider == "azure":
            return bool(os.getenv("AZURE_OPENAI_API_KEY") and self.azure_endpoint)
        if self.provider == "ollama":
            return True
        return False

    def _chat_with_llm(self, user_message: str) -> CopilotResponse:
        tool_spec_str = json.dumps(self.tools.list_tools(), indent=2)
        sys_prompt = f"{SYSTEM_PROMPT}\n\n{TOOL_CALLING_INSTRUCTION}\nTools:\n{tool_spec_str}"
        messages = [{"role": "system", "content": sys_prompt}, *self.memory.to_messages()]

        raw_text = self._call_llm_provider(messages)
        tools_executed: list[dict[str, Any]] = []

        # Check if LLM returned a JSON tool call
        parsed_call = self._extract_json_tool_call(raw_text)
        if parsed_call is not None:
            t_name = str(parsed_call["tool"])
            t_args = dict(parsed_call.get("arguments", {}))
            result = self.tools.invoke(t_name, t_args)
            tools_executed.append({"tool": t_name, "arguments": t_args, "result": result})
            self.memory.add_turn("tool", json.dumps(result), tool_name=t_name, tool_result=result)
            followup = [
                *messages,
                {"role": "assistant", "content": raw_text},
                {"role": "user", "content": f"Tool '{t_name}' returned:\n{json.dumps(result)}\nSummarize the key insights in English and recommend next steps."},
            ]
            final_reply = self._call_llm_provider(followup)
            return CopilotResponse(
                reply=final_reply,
                provider_used=self.provider,
                tools_executed=tools_executed,
                suggested_next_steps=self._default_next_steps(t_name),
            )

        return CopilotResponse(
            reply=raw_text,
            provider_used=self.provider,
            tools_executed=tools_executed,
            suggested_next_steps=["Ask me to profile data quality, run AutoML, explain predictions, or export a pipeline."],
        )

    def _call_llm_provider(self, messages: list[dict[str, str]]) -> str:
        with httpx.Client(timeout=45.0) as client:
            if self.provider == "openai":
                resp = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                    json={"model": self.model, "temperature": self.temperature, "messages": messages},
                )
                resp.raise_for_status()
                return str(resp.json()["choices"][0]["message"]["content"])

            if self.provider == "azure":
                resp = client.post(
                    f"{str(self.azure_endpoint).rstrip('/')}/openai/deployments/{self.model}/chat/completions?api-version=2024-02-01",
                    headers={"api-key": os.environ["AZURE_OPENAI_API_KEY"]},
                    json={"temperature": self.temperature, "messages": messages},
                )
                resp.raise_for_status()
                return str(resp.json()["choices"][0]["message"]["content"])

            if self.provider == "anthropic":
                sys_content = "\n".join(m["content"] for m in messages if m["role"] == "system")
                non_sys = [m for m in messages if m["role"] != "system"]
                resp = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                        "anthropic-version": "2023-06-01",
                    },
                    json={"model": self.model, "max_tokens": 1024, "system": sys_content, "messages": non_sys},
                )
                resp.raise_for_status()
                return str(resp.json()["content"][0]["text"])

            # Ollama local REST endpoint
            resp = client.post(
                f"{self.ollama_base_url.rstrip('/')}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            return str(resp.json()["message"]["content"])

    def _chat_local_planner(self, user_message: str) -> CopilotResponse:
        """Deterministic English NLP planner routing instructions directly to live OmniForge tools.

        Uses word-boundary keyword matching (not substring matching) so that short
        keywords like ``ate`` do not false-positive on ``validate``, ``estimate``,
        or ``Update``. Never silently generates demo data — requires an explicitly
        loaded dataset for data-dependent intents.
        """
        msg = user_message.lower().strip()
        executed: list[dict[str, Any]] = []
        reply_parts: list[str] = []
        next_steps: list[str] = []

        def _contains_word(text: str, keywords: tuple[str, ...]) -> bool:
            """Return True if any keyword appears as a whole word in ``text``."""
            return any(re.search(r"\b" + re.escape(k) + r"\b", text) for k in keywords)

        class _NoDatasetError(Exception):
            pass

        def _require_dataset() -> None:
            """Require an explicitly loaded dataset; never silently generate demo data."""
            if not self.memory.working_state["datasets"]:
                reply_parts.append(
                    "No dataset is loaded. Please load a dataset first (e.g., upload a CSV or "
                    "call ``load_dataframe``), then repeat your request."
                )
                raise _NoDatasetError()

        # Extract optional target column from patterns like "target=y" or "predict churn"
        target_match = re.search(r"target[=:\s]+['\"]?([A-Za-z0-9_]+)['\"]?", user_message, re.I)
        target_col = target_match.group(1) if target_match else "target"

        try:
            if _contains_word(msg, ("domain", "industry", "plugin")):
                res = self.tools.invoke("list_domains")
                executed.append({"tool": "list_domains", "arguments": {}, "result": res})
                names = [d["display_name"] for d in res["domains"]]
                reply_parts.append(f"OmniForge includes **{len(names)} ready-to-use industry domain plugins**: {', '.join(names)}.")
                next_steps = ["Specify a domain (e.g., finance, healthcare, energy) to load domain KPIs and compliance rules."]

            elif _contains_word(msg, ("pipeline", "code", "dag", "generate pipeline")):
                res = self.tools.invoke("generate_pipeline_code", {"target_column": target_col})
                executed.append({"tool": "generate_pipeline_code", "arguments": {"target_column": target_col}, "result": res})
                reply_parts.append(
                    f"Generated a reproducible DAG pipeline (`{res['pipeline_name']}`) with Mermaid topology and standalone Python code:\n\n"
                    f"```mermaid\n{res['mermaid_dag']}\n```"
                )
                next_steps = ["Execute the pipeline via PipelineExecutor or export to Jupyter Notebook."]

            elif _contains_word(msg, ("forecast", "time series", "time-series")):
                _require_dataset()
                _, df = self.memory.get_dataset()
                fc_target = target_col if target_col in df.columns else ("y" if "y" in df.columns else str(df.select_dtypes(include=["number"]).columns[-1]))
                res = self.tools.invoke("run_forecast", {"target_column": fc_target, "horizon": 14})
                executed.append({"tool": "run_forecast", "arguments": {"target_column": fc_target, "horizon": 14}, "result": res})
                reply_parts.append(
                    f"Completed forecasting tournament for **`{fc_target}`** (horizon=14). "
                    f"Champion model: **{res['champion_model']}** (top leaderboard entry: `{res['leaderboard'][0]}`)."
                )
                next_steps = ["Run walk-forward backtesting", "Export interactive HTML/PDF forecast report"]

            elif _contains_word(msg, ("causal", "treatment", "uplift")):
                _require_dataset()
                res = self.tools.invoke("run_causal", {"method": "aipw"})
                executed.append({"tool": "run_causal", "arguments": {"method": "aipw"}, "result": res})
                reply_parts.append(
                    f"Doubly Robust (AIPW) causal estimate: **ATE = {res['ate']}** "
                    f"(95% CI: [{res['ci_lower']}, {res['ci_upper']}], p={res['p_value']}). "
                    f"Refutation checks: `{res['refutations']}`."
                )
                next_steps = ["Compare against Double Machine Learning (DML)", "Inspect heterogeneous CATE subgroups"]

            elif _contains_word(msg, ("clean", "impute", "outlier", "dedup")):
                _require_dataset()
                res = self.tools.invoke("clean_data")
                executed.append({"tool": "clean_data", "arguments": {}, "result": res})
                reply_parts.append(
                    f"Data cleaning completed into **`{res['output_dataset']}`**: "
                    f"missing values reduced from {res['missing_values_before']} to {res['missing_values_after']}, "
                    f"exact duplicates removed={res['exact_duplicates_removed']}, outliers handled={res['outliers_detected']}."
                )
                next_steps = ["Run Automated EDA", "Train AutoML models with baseline benchmarks"]

            elif _contains_word(msg, ("quality", "scorecard")):
                _require_dataset()
                res = self.tools.invoke("profile_quality")
                executed.append({"tool": "profile_quality", "arguments": {}, "result": res})
                reply_parts.append(
                    f"Six-dimensional Data Quality Score: **{res['overall_score']}/100 (Grade {res['grade']})**. "
                    f"Dimensions: {res['dimensions']}. Recommendations: {'; '.join(res['recommendations'])}."
                )
                next_steps = ["Clean detected data issues", "Auto-generate a DataValidator expectation suite"]

            elif _contains_word(msg, ("eda", "explore", "profile", "insight")):
                _require_dataset()
                res = self.tools.invoke("run_eda", {"target_column": target_col})
                executed.append({"tool": "run_eda", "arguments": {"target_column": target_col}, "result": res})
                reply_parts.append(res["narrative"])
                next_steps = ["Clean data if alerts were raised", "Run AutoML pipeline search"]

            elif _contains_word(msg, ("fairness", "bias", "disparate")):
                _require_dataset()
                if not self.memory.working_state["models"]:
                    tr = self.tools.invoke("train_automl", {"target_column": target_col, "time_budget_seconds": 15.0})
                    executed.append({"tool": "train_automl", "arguments": {"target_column": target_col}, "result": tr})
                res = self.tools.invoke("audit_fairness", {"target_column": target_col})
                executed.append({"tool": "audit_fairness", "arguments": {"target_column": target_col}, "result": res})
                reply_parts.append(
                    f"Fairness Audit for `{res['protected_attribute']}`: **{'PASSED' if res['passed'] else 'FLAGGED'}** "
                    f"(Disparate Impact={res['disparate_impact_ratio']}, Equalized Odds Diff={res['equalized_odds_diff']})."
                )
                next_steps = ["Apply Kamiran-Calders reweighing if flagged", "Generate a compliance Model Card"]

            elif _contains_word(msg, ("explain", "shap", "interpret", "why")):
                _require_dataset()
                if not self.memory.working_state["models"]:
                    tr = self.tools.invoke("train_automl", {"target_column": target_col, "time_budget_seconds": 15.0})
                    executed.append({"tool": "train_automl", "arguments": {"target_column": target_col}, "result": tr})
                res = self.tools.invoke("explain_model", {"target_column": target_col})
                executed.append({"tool": "explain_model", "arguments": {"target_column": target_col}, "result": res})
                top_shap = list(res["local_shap_sample_0"]["attributions"].items())[:5]
                reply_parts.append(
                    f"Explainability report for **{res['model']}**: top local SHAP drivers on sample 0 are "
                    + ", ".join(f"`{k}` ({v:+.4f})" for k, v in top_shap)
                    + "."
                )
                next_steps = ["Run fairness audit", "Run adversarial robustness stress test"]

            elif _contains_word(msg, ("train", "automl", "model", "predict")):
                _require_dataset()
                res = self.tools.invoke("train_automl", {"target_column": target_col, "time_budget_seconds": 25.0})
                executed.append({"tool": "train_automl", "arguments": {"target_column": target_col}, "result": res})
                reply_parts.append(
                    f"AutoML complete for `{res['task_type']}`! Champion: **{res['best_algorithm']}** "
                    f"achieving **{res['primary_metric']} = {res['best_score']}** "
                    f"(lift over strongest baseline: **{res['lift_over_best_baseline']:+.4f}**; "
                    f"carbon footprint: {res['carbon_g_co2eq']:.6f} gCO2eq)."
                )
                next_steps = ["Explain model with SHAP/LIME", "Audit fairness & robustness", "Register model to Production"]

            elif _contains_word(msg, ("root cause", "root-cause", "drift attribution", "causal drift")):
                _require_dataset()
                ref_name = "reference_dataset"
                cur_name = "current_dataset"
                if ref_name not in self.memory.working_state["datasets"]:
                    _require_dataset()
                if cur_name not in self.memory.working_state["datasets"]:
                    _require_dataset()
                res = self.tools.invoke("root_cause_diagnosis", {"reference_dataset": ref_name, "current_dataset": cur_name})
                executed.append({"tool": "root_cause_diagnosis", "arguments": {"reference_dataset": ref_name, "current_dataset": cur_name}, "result": res})
                rc = res.get("root_causes", [])
                sym = res.get("symptoms", [])
                rx = res.get("remediation_prescriptions", [])
                reply_parts.append(
                    f"C5 Causal Drift Root-Cause (status={res.get('overall_status')}): "
                    f"root causes={rc or 'none isolated'}, symptoms={sym or 'none'}. "
                    + (" " + " ".join(rx[:3]) if rx else "")
                )
                next_steps = ["Remediate upstream root-cause sensors/feeds", "Re-run drift check after remediation", "Register updated model"]

            elif _contains_word(msg, ("twin control", "mpc", "closed-loop", "closed loop", "receding horizon", "digital twin control")):
                res = self.tools.invoke("twin_control", {"n_x": 2, "n_u": 1, "horizon": 10, "n_steps": 8, "dt": 1.0})
                executed.append({"tool": "twin_control", "arguments": {"n_x": 2, "n_u": 1, "horizon": 10, "n_steps": 8, "dt": 1.0}, "result": res})
                reply_parts.append(
                    f"C7 Closed-Loop Twin MPC: {res['n_steps_applied']} steps applied, "
                    f"constraints satisfied={res['all_constraints_satisfied']}, total cost={res['total_cost']:.4f}."
                )
                next_steps = ["Inspect trajectory chart", "Tighten state safety bounds", "Export control log"]

            elif _contains_word(msg, ("split lock", "split_lock", "leakage guard", "train/test integrity")):
                res = self.tools.invoke("split_lock", {})
                executed.append({"tool": "split_lock", "arguments": {}, "result": res})
                passed = res.get("passed", False)
                reply_parts.append(
                    f"Split Lock: {'PASSED' if passed else 'FAILED'} — "
                    f"row_overlap={res.get('row_overlap_count', 0)}, group_overlap={res.get('group_overlap_count', 0)}, "
                    f"temporal_valid={res.get('temporal_order_valid', True)}."
                )
                next_steps = ["Fix overlapping groups/rows before training", "Re-run split lock"]

            elif _contains_word(msg, ("pii audit", "pii_audit", "pseudonymize", "pii guard", "zero trust pii")):
                res = self.tools.invoke("pii_audit", {"dataset_name": None, "pii_policy": "pseudonymize"})
                executed.append({"tool": "pii_audit", "arguments": {"dataset_name": None, "pii_policy": "pseudonymize"}, "result": res})
                reply_parts.append(
                    f"C8 PII Audit (policy=pseudonymize): PII detected={res.get('pii_detected')}, "
                    f"columns={', '.join(res.get('pii_columns', [])) or 'none'}, cells={res.get('pii_cells_total', 0)}."
                )
                next_steps = ["Inspect guarded output dataset", "Re-audit with policy=block to fail-closed"]

            elif _contains_word(msg, ("pareto", "green-ai", "carbon budget", "latency budget", "green automl")):
                res = self.tools.invoke("pareto_automl", {"target_column": target_col, "time_budget_seconds": 25.0})
                executed.append({"tool": "pareto_automl", "arguments": {"target_column": target_col, "time_budget_seconds": 25.0}, "result": res})
                reply_parts.append(
                    f"C3 Green-AI Pareto AutoML: champion={res.get('best_algorithm')} "
                    f"({res.get('primary_metric')}={res.get('best_score')}, "
                    f"carbon={res.get('carbon_gco2e'):.6f} gCO2eq). Pareto frontier: {len(res.get('pareto_frontier', []))} candidates."
                )
                next_steps = ["Inspect Pareto frontier trade-offs", "Register Pareto champion", "Promote to Production"]

            elif _contains_word(msg, ("domain validate", "domain_validation", "industry domain", "compliance")):
                domain_name = target_col
                res = self.tools.invoke("domain_validate", {"domain": domain_name, "dataset_name": None})
                executed.append({"tool": "domain_validate", "arguments": {"domain": domain_name, "dataset_name": None}, "result": res})
                reply_parts.append(f"Domain validation for '{domain_name}': {res.get('note', 'checks completed')}.")
                next_steps = ["Address domain compliance gaps", "Generate Model Card"]

            elif _contains_word(msg, ("register model", "model registry", "promote model", "model promotion")):
                res = self.tools.invoke("register_model", {"model_name": "copilot_champion", "stage": "Staging"})
                executed.append({"tool": "register_model", "arguments": {"model_name": "copilot_champion", "stage": "Staging"}, "result": res})
                reply_parts.append(f"Model registered to MLOps Registry (Staging): {res.get('registered', {}).get('version', 'n/a')}.")
                next_steps = ["Promote to Production", "Configure drift monitoring"]

            elif _contains_word(msg, ("list studios", "studios", "studio overview")):
                res = self.tools.invoke("list_studios", {})
                executed.append({"tool": "list_studios", "arguments": {}, "result": res})
                studio_names = ", ".join(s["name"] for s in res.get("studios", []))
                reply_parts.append(f"OmniForge Workbench has {res.get('count', 0)} studios: {studio_names}.")
                next_steps = ["Open a studio in the Workbench UI", "Run a studio endpoint via API"]

            elif _contains_word(msg, ("workspace", "status")):
                res = self.tools.invoke("workspace_status", {})
                executed.append({"tool": "workspace_status", "arguments": {}, "result": res})
                reply_parts.append(
                    f"Current workspace state: {len(res.get('loaded_datasets', []))} dataset(s), {len(res.get('trained_models', []))} trained model(s)."
                )
                next_steps = ["Profile data quality", "Run AutoML", "Generate production pipeline code"]

            else:
                reply_parts.append(
                    "I'm not sure what you'd like to do. I can help you with:\n"
                    "• Profile data quality, clean data, run Automated EDA\n"
                    "• Train AutoML models, explain predictions (SHAP/LIME), audit fairness\n"
                    "• Run forecasting or causal inference (ATE, DML, DiD)\n"
                    "• Enforce split lock, audit PII, run Green-AI Pareto AutoML\n"
                    "• Validate against industry domains, list studios, export pipeline code\n\n"
                    "Please load a dataset first if your request involves data analysis."
                )
                next_steps = ["Load a dataset", "Profile data quality", "Run AutoML", "Generate production pipeline code"]

        except _NoDatasetError:
            # reply_parts already contains the "no dataset loaded" message from _require_dataset()
            pass

        return CopilotResponse(
            reply="\n\n".join(reply_parts),
            provider_used="local_deterministic_planner",
            tools_executed=executed,
            suggested_next_steps=next_steps,
        )

    @staticmethod
    def _extract_json_tool_call(text: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\"tool\".*\}", text, re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "tool" in obj:
                return obj
        except json.JSONDecodeError:
            return None
        return None

    @staticmethod
    def _default_next_steps(last_tool: str) -> list[str]:
        mapping = {
            "ingest_data": ["Profile 6-dimensional data quality", "Run Automated EDA"],
            "profile_quality": ["Clean missing values and outliers", "Run validation suite"],
            "clean_data": ["Run Automated EDA", "Train AutoML models"],
            "train_automl": ["Explain predictions with SHAP", "Audit fairness", "Register to Model Registry"],
        }
        return mapping.get(last_tool, ["Generate executive report", "Export pipeline code"])
