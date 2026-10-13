import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
RANKER_CONFIG_PATH = BASE_DIR / "ranker_config.json"
UI_STATE_PATH = BASE_DIR / "ui_state.json"
SNAPSHOT_PATH = BASE_DIR / "jobs_snapshot.json"
RANKED_JOBS_PATH = BASE_DIR / "ranked_jobs.json"
LABELS_PATH = BASE_DIR / "labels.json"
CSV_PATH = BASE_DIR / "jobs_for_labelling.csv"
CV_PATH = BASE_DIR / "resume" / "cv.pdf"

_DEFAULT_CONFIG: dict = {
    "w_skills": 0.75,
    "w_seniority": 0.00,
    "w_location": 0.25,
    "w_company": 0.00,
    "w_salary_bonus": 0.00,
    "top_k": 15,
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "judge_prompt": (
        "Score job-candidate fit on a 1-10 scale per dimension. "
        "Be strict: 10=perfect, 5=average, 1=poor. "
        "Omit salary_fit if salary is not listed in the posting."
    ),
}
_EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
_LLM_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]

# Lines from subprocesses that are noise — Opik trace URLs, Pydantic serialiser warnings
_NOISE_PATTERNS = [
    "OPIK: Started logging traces",
    "UserWarning: Pydantic serializer",
    "PydanticSerializationUnexpectedValue",
    "return self.__pydantic_serializer__",
    "pydantic\\main.py",
    "pydantic/main.py",
    "  warnings.warn(",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if RANKER_CONFIG_PATH.exists():
        return {**_DEFAULT_CONFIG, **json.loads(RANKER_CONFIG_PATH.read_text())}
    return _DEFAULT_CONFIG.copy()


def _save_config(config: dict) -> None:
    RANKER_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def _load_ui_state() -> dict:
    defaults: dict = {
        "labels_stale": False,
        "last_metrics": None,
        "last_experiment": "",
        "last_config_key": "",
    }
    if UI_STATE_PATH.exists():
        return {**defaults, **json.loads(UI_STATE_PATH.read_text())}
    return defaults


def _save_ui_state(state: dict) -> None:
    UI_STATE_PATH.write_text(json.dumps(state, indent=2))


def _config_key(config: dict) -> str:
    sig = config["judge_prompt"] + config["llm_model"]
    return hashlib.md5(sig.encode()).hexdigest()[:8]


def _should_show(line: str) -> bool:
    return not any(p in line for p in _NOISE_PATTERNS)


def _run_with_progress(cmd: list[str], label: str, estimated_seconds: int = 60) -> tuple[int, str]:
    """Run a subprocess, stream filtered output, and show a time-based progress bar."""
    progress_bar = st.progress(0.0, text=label)
    output_box = st.empty()
    start = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(BASE_DIR),
        env=env,
        bufsize=1,
    )
    output = ""
    for raw_line in proc.stdout:  # type: ignore[union-attr]
        if _should_show(raw_line):
            output += raw_line
            output_box.code(output, language="")
        elapsed = time.time() - start
        progress_bar.progress(
            min(0.95, elapsed / estimated_seconds),
            text=f"{label} ({int(elapsed)}s elapsed)",
        )
    proc.wait()
    elapsed_total = time.time() - start
    progress_bar.progress(1.0, text=f"Done ({int(elapsed_total)}s)")
    return proc.returncode, output


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Job Search Agent", layout="wide")
st.title("Job Search Agent")

saved_config = _load_config()
ui_state = _load_ui_state()

# ── Sidebar: Ranker Config ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Ranker Config")

    w_skills      = st.slider("w_skills",      0.0, 1.0, float(saved_config["w_skills"]),      0.05)
    w_seniority   = st.slider("w_seniority",   0.0, 1.0, float(saved_config["w_seniority"]),   0.05)
    w_location    = st.slider("w_location",    0.0, 1.0, float(saved_config["w_location"]),    0.05)
    w_company     = st.slider("w_company",     0.0, 1.0, float(saved_config["w_company"]),     0.05)
    w_salary_bonus = st.slider("w_salary_bonus (additive)", 0.0, 0.5, float(saved_config["w_salary_bonus"]), 0.05)

    top_k = st.number_input("top_k", min_value=5, max_value=50, value=int(saved_config["top_k"]), step=1)

    emb_idx = _EMBEDDING_MODELS.index(saved_config["embedding_model"]) if saved_config["embedding_model"] in _EMBEDDING_MODELS else 0
    embedding_model = st.selectbox("embedding_model", _EMBEDDING_MODELS, index=emb_idx)

    llm_idx = _LLM_MODELS.index(saved_config["llm_model"]) if saved_config["llm_model"] in _LLM_MODELS else 0
    llm_model = st.selectbox("llm_model", _LLM_MODELS, index=llm_idx)

    judge_prompt = st.text_area("judge_prompt", value=saved_config["judge_prompt"], height=150)

    # Warning: judge prompt or model edited but not yet saved
    llm_behavior_changed = (
        judge_prompt != saved_config["judge_prompt"]
        or llm_model != saved_config["llm_model"]
    )
    if llm_behavior_changed:
        st.warning(
            "Changing judge prompt or model will trigger new LLM calls (~2 min, will cost money). "
            "Weight-only changes are instant and free."
        )

    weight_sum = round(w_skills + w_seniority + w_location + w_company, 10)
    weights_valid = abs(weight_sum - 1.0) <= 0.005
    if not weights_valid:
        st.error(
            f"w_skills + w_seniority + w_location + w_company = {weight_sum:.2f} — must equal 1.0. "
            "w_salary_bonus is additive and excluded from this sum."
        )

    current_form_config: dict = {
        "w_skills": w_skills,
        "w_seniority": w_seniority,
        "w_location": w_location,
        "w_company": w_company,
        "w_salary_bonus": w_salary_bonus,
        "top_k": int(top_k),
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "judge_prompt": judge_prompt,
    }
    has_unsaved = current_form_config != saved_config
    if has_unsaved and not llm_behavior_changed:
        st.info("Unsaved changes.")

    if st.button("Save Config", use_container_width=True, disabled=not weights_valid):
        old_key = _config_key(saved_config)
        new_key = _config_key(current_form_config)
        _save_config(current_form_config)
        # Tell the Jobs tab to re-rank with the right mode
        if SNAPSHOT_PATH.exists():
            st.session_state.rerank_requested = True
            st.session_state.rerank_reweight_only = (old_key == new_key)
        st.success("Config saved.")
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_jobs, tab_eval, tab_label = st.tabs(["Jobs", "Eval", "Label"])

# ── Jobs tab ──────────────────────────────────────────────────────────────────
with tab_jobs:

    # Auto-rerank when triggered by Save Config
    if st.session_state.get("rerank_requested") and SNAPSHOT_PATH.exists():
        reweight_only = st.session_state.get("rerank_reweight_only", False)
        st.session_state.rerank_requested = False  # clear before running to avoid loops
        cmd = ["uv", "run", "python", "rerank.py"]
        if reweight_only:
            cmd.append("--reweight-only")
        est = 3 if reweight_only else 75
        rc, _ = _run_with_progress(cmd, "Refreshing job rankings...", estimated_seconds=est)
        if rc != 0:
            st.error("Rerank failed. Check output above.")
        else:
            st.rerun()

    if RANKED_JOBS_PATH.exists():
        jobs: list[dict] = json.loads(RANKED_JOBS_PATH.read_text())
        st.caption(f"{len(jobs)} jobs ranked")

        # ── Sortable table ────────────────────────────────────────────────────
        df = pd.DataFrame([
            {
                "#": i + 1,
                "Title": j["title"],
                "Company": j["company"],
                "Location": j["location"],
                "Remote": bool(j.get("is_remote", False)),
                "Skills": int(j.get("skills_match", 0)),
                "Loc.Fit": int(j.get("location_fit", 0)),
                "Score": float(j.get("final_score", 0.0)),
                "Link": j["url"],
            }
            for i, j in enumerate(jobs)
        ])
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Remote": st.column_config.CheckboxColumn(width="small"),
                "Skills": st.column_config.ProgressColumn(
                    min_value=0, max_value=10, format="%d/10", width="medium"
                ),
                "Loc.Fit": st.column_config.ProgressColumn(
                    min_value=0, max_value=10, format="%d/10", width="medium"
                ),
                "Score": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.3f", width="medium"
                ),
                "Link": st.column_config.LinkColumn(display_text="Apply ↗", width="small"),
            },
        )

        # ── Detail card ───────────────────────────────────────────────────────
        st.divider()
        options = [f"{i + 1}. {j['title']} @ {j['company']}" for i, j in enumerate(jobs)]
        selected = st.selectbox(
            "View job details",
            options=options,
            index=None,
            placeholder="Select a job to see full details...",
        )

        if selected is not None:
            idx = options.index(selected)
            j = jobs[idx]
            with st.container(border=True):
                col_hd, col_btn = st.columns([5, 1])
                col_hd.subheader(j["title"])
                col_btn.link_button("Apply ↗", j["url"], use_container_width=True)

                meta = [j["company"], j["location"]]
                if j.get("is_remote"):
                    meta.append("Remote")
                if j.get("employment_type"):
                    meta.append(j["employment_type"])
                st.caption(" · ".join(meta))

                if j.get("salary_min"):
                    st.caption(
                        f"Salary: {j.get('salary_currency', '')} "
                        f"{j['salary_min']}–{j.get('salary_max', '')}"
                    )

                st.markdown("**Score breakdown**")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Skills",   f"{j.get('skills_match', '—')}/10")
                c2.metric("Seniority",f"{j.get('seniority_alignment', '—')}/10")
                c3.metric("Location", f"{j.get('location_fit', '—')}/10")
                c4.metric("Company",  f"{j.get('company_score', '—')}/10")
                c5.metric("Final",    f"{j.get('final_score', 0):.3f}")

                if j.get("reasoning"):
                    st.info(j["reasoning"])

                if j.get("posted_date"):
                    st.caption(f"Posted: {j['posted_date']}")

        st.divider()
    else:
        st.info("No jobs yet. Run the agent below to fetch and rank jobs.")

    # ── Get New Jobs (always available, expanded when no jobs exist) ───────────
    with st.expander("Get New Jobs", expanded=not RANKED_JOBS_PATH.exists()):
        st.write("**Upload new CV (optional — leave blank to use existing)**")
        uploaded_cv = st.file_uploader("CV PDF", type=["pdf"], label_visibility="collapsed")

        if uploaded_cv is not None:
            cv_key = f"{uploaded_cv.name}_{uploaded_cv.size}"
            if st.session_state.get("last_cv_key") != cv_key:
                CV_PATH.parent.mkdir(parents=True, exist_ok=True)
                CV_PATH.write_bytes(uploaded_cv.read())
                ui_state["labels_stale"] = True
                _save_ui_state(ui_state)
                st.session_state.last_cv_key = cv_key
                st.success("CV saved to resume/cv.pdf")

        if CV_PATH.exists():
            st.caption(f"Current CV: resume/cv.pdf  ({CV_PATH.stat().st_size // 1024} KB)")
        else:
            st.warning("No CV found at resume/cv.pdf")

        if ui_state.get("labels_stale"):
            st.info("CV or snapshot updated. After running the agent, go to the Label tab to re-label.")

        if st.button("Run Agent Now", disabled=not CV_PATH.exists()):
            rc, _ = _run_with_progress(
                ["uv", "run", "python", "agent.py", "--snapshot"],
                "Fetching fresh jobs from JSearch...",
                estimated_seconds=75,
            )
            if rc == 0:
                ui_state["labels_stale"] = True
                _save_ui_state(ui_state)
                # Trigger fresh rerank of new snapshot
                st.session_state.rerank_requested = True
                st.session_state.rerank_reweight_only = False
                st.rerun()
            else:
                st.error("Agent run failed. Check output above.")

# ── Eval tab ──────────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Eval Metrics")

    if ui_state.get("last_metrics"):
        m = ui_state["last_metrics"]
        col1, col2 = st.columns(2)
        col1.metric("P@10", f"{m.get('precision_at_10', 0):.3f}")
        col2.metric("NDCG@10", f"{m.get('ndcg_at_10', 0):.3f}")
        if ui_state.get("last_experiment"):
            st.caption(f"Last run: {ui_state['last_experiment']}")
    else:
        st.info("No eval results yet. Run eval below.")

    st.divider()

    last_key = ui_state.get("last_config_key", "")
    current_key = _config_key(saved_config)
    if last_key and current_key != last_key:
        st.warning(
            "Judge prompt or model changed since last eval run — "
            "this will trigger new LLM calls for uncached jobs (~2 min, will cost money)."
        )
    if has_unsaved:
        st.info("You have unsaved config changes. Save config in the sidebar — eval uses the saved config.")
    if ui_state.get("labels_stale"):
        st.warning(
            "A new CV or snapshot was loaded since the last labeling run. "
            "New jobs won't appear in metrics until labeled. "
            "Go to the Label tab to update labels, or run eval now to score existing labeled jobs only."
        )

    eval_blocked = False
    if not SNAPSHOT_PATH.exists():
        st.error("No snapshot found. Run the agent from the Jobs tab first.")
        eval_blocked = True
    elif not LABELS_PATH.exists():
        st.error("No labels found. Export CSV from the Label tab, fill it in, and import.")
        eval_blocked = True

    experiment_name = st.text_input(
        "Experiment name",
        value=ui_state.get("last_experiment", ""),
        placeholder="e.g. high_skills_no_seniority",
    )

    if st.button("Run Eval", disabled=eval_blocked):
        if not experiment_name.strip():
            st.error("Enter an experiment name.")
        else:
            rc, output = _run_with_progress(
                ["uv", "run", "python", "eval.py", "--experiment-name", experiment_name.strip()],
                "Running eval...",
                estimated_seconds=120,
            )
            if rc == 0:
                metrics: dict[str, float] = {}
                for line in output.splitlines():
                    parts = line.strip().split()
                    if len(parts) == 2 and "_" in parts[0]:
                        try:
                            metrics[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
                ui_state["last_metrics"] = {
                    "precision_at_10": metrics.get("precision_at_10", 0.0),
                    "ndcg_at_10": metrics.get("ndcg_at_10", 0.0),
                }
                ui_state["last_experiment"] = experiment_name.strip()
                ui_state["last_config_key"] = current_key
                _save_ui_state(ui_state)
                st.success("Eval complete!")
                st.rerun()
            else:
                st.error("Eval failed. Check output above.")

# ── Label tab ─────────────────────────────────────────────────────────────────
with tab_label:
    st.subheader("Label Jobs")

    col1, col2 = st.columns(2)
    snapshot_count = len(json.loads(SNAPSHOT_PATH.read_text())) if SNAPSHOT_PATH.exists() else None
    labels_count = len(json.loads(LABELS_PATH.read_text())) if LABELS_PATH.exists() else None
    col1.metric("Jobs in snapshot", snapshot_count if snapshot_count is not None else "—")
    col2.metric("Labeled jobs", labels_count if labels_count is not None else "—")

    if ui_state.get("labels_stale"):
        st.warning("Labels are stale. Export a fresh CSV, fill in the relevance column, and import below.")

    st.divider()
    st.write("**Step 1 — Export CSV for labeling**")
    st.caption("Re-running export preserves existing labels in the relevance column (pre-filled).")

    if st.button("Export CSV", disabled=not SNAPSHOT_PATH.exists()):
        rc, _ = _run_with_progress(
            ["uv", "run", "python", "label_jobs.py", "--export"],
            "Exporting CSV...",
            estimated_seconds=5,
        )
        if rc != 0:
            st.error("Export failed. Check output above.")

    if CSV_PATH.exists():
        st.download_button(
            label="Download jobs_for_labelling.csv",
            data=CSV_PATH.read_bytes(),
            file_name="jobs_for_labelling.csv",
            mime="text/csv",
        )

    st.divider()
    st.write("**Step 2 — Upload labeled CSV and import**")
    st.caption("Fill the `relevance` column: 0 = not relevant, 1 = somewhat, 2 = highly relevant.")

    uploaded_labeled = st.file_uploader("Upload labeled CSV", type=["csv"], key="labeled_csv")
    if uploaded_labeled is not None:
        labeled_key = f"{uploaded_labeled.name}_{uploaded_labeled.size}"
        if st.session_state.get("last_labeled_key") != labeled_key:
            CSV_PATH.write_bytes(uploaded_labeled.read())
            st.session_state.last_labeled_key = labeled_key
            st.success("CSV uploaded.")

    if CSV_PATH.exists():
        if st.button("Import Labels"):
            rc, _ = _run_with_progress(
                ["uv", "run", "python", "label_jobs.py", "--import"],
                "Importing labels...",
                estimated_seconds=3,
            )
            if rc == 0:
                ui_state["labels_stale"] = False
                _save_ui_state(ui_state)
                st.success("Labels imported. You can now run eval from the Eval tab.")
                st.rerun()
            else:
                st.error("Import failed. Check output above.")
    else:
        st.info("Export a CSV first, then upload the labeled version here.")
