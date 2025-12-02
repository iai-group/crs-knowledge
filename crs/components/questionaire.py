import json
import logging
import os
from datetime import datetime

import requests
import streamlit as st

logger = logging.getLogger(__name__)

QUESTIONNAIRE_DIR = "data/questionnaires/"
MAX_GROUP_PARTICIPANTS = 50


def _ensure_output_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_screen_response(
    answers: dict,
    assigned_domain: str = None,
    assigned_expertise: str = None,
    out_path: str = "exports/screen_results.jsonl",
) -> None:
    _ensure_output_dir(out_path)
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "answers": answers,
        "assigned_domain": assigned_domain,
        "assigned_expertise": assigned_expertise,
        "prolific": st.session_state.get("prolific", {}),
    }
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        st.warning(f"Failed to save response to {out_path}: {e}")


def count_participants_by_domain_expertise(
    out_path: str = "exports/screen_results.jsonl",
) -> dict:
    """
    Count participants by their assigned domain-expertise combination.
    Only counts participants who completed the study.

    Returns:
        A dictionary with keys like "bicycle-novice", "digital_camera-expert", etc.
        mapping to their completed participant counts.
    """
    counts_path = os.path.join("exports", "screen_counts.json")
    if os.path.exists(counts_path):
        try:
            with open(counts_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.warning(
                "Failed to load precomputed counts from %s: %s; falling back to raw log counting",
                counts_path,
                e,
            )

    counts = {}
    if not os.path.exists(out_path):
        return counts

    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                assigned_domain = rec.get("assigned_domain", None)
                assigned_expertise = rec.get("assigned_expertise", None)

                if assigned_domain and assigned_expertise:
                    count_key = (
                        f"{assigned_domain}-{assigned_expertise.lower()}"
                    )
                    counts[count_key] = counts.get(count_key, 0) + 1
    except Exception as e:
        logger.error(f"Error counting participants: {e}")
        return counts

    return counts


def assign_domain_to_participant(
    participant_answers: dict,
    out_path: str = "exports/screen_results.jsonl",
) -> tuple[str, str]:
    """
    Assign a domain to a participant based on balancing participants across domain-expertise combinations.

    Strategy:
    1. Find the domain-expertise combination with the fewest assigned participants
    2. On ties, prioritize higher expertise levels (Expert > Intermediate > Novice)

    Args:
        participant_answers: Dictionary with internal domain names as keys and knowledge levels as values
                           e.g., {"movie": "Expert", "bicycle": "Novice", "digital_camera": "Intermediate", ...}
        out_path: Path to screen_results.jsonl file

    Returns:
        A tuple of (domain_name, expertise_level) in internal format
        e.g., ("bicycle", "Novice") or ("digital_camera", "Expert")
    """
    counts = count_participants_by_domain_expertise(out_path)

    available_domains = st.session_state.get("domains", [])

    expertise_rank = {
        "expert": 0,
        "intermediate": 2,
        "novice": 1,
    }

    candidates = []

    for domain_key, knowledge_level in participant_answers.items():
        if not knowledge_level or knowledge_level == "":
            continue

        if domain_key not in available_domains:
            continue

        count_key = f"{domain_key}-{knowledge_level.lower()}"
        count = counts.get(count_key, 0)

        expertise_priority = expertise_rank.get(knowledge_level.lower(), 999)

        candidates.append(
            (count, expertise_priority, domain_key, knowledge_level)
        )

    if not candidates:
        logger.warning(
            "No valid domain answers found for available domains, defaulting to bicycle-Novice"
        )
        return ("bicycle", "Novice")

    candidates.sort()

    assigned_domain = candidates[0][2]
    assigned_expertise = candidates[0][3]
    logger.info(
        f"Assigned domain-expertise: {assigned_domain}-{assigned_expertise} (count: {candidates[0][0]}, expertise_rank: {candidates[0][1]})"
    )

    return (assigned_domain, assigned_expertise)


def load_questionnaire(directory: str, name: str) -> list:
    """
    If needed, load additional questions for 'post' or other questionnaires
    from JSON. For 'pre', we do a custom pairwise question, so we might not
    even need to load a file.
    """
    questions = []
    filepath = os.path.join(directory, f"{name}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    questions.extend(data)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    return questions


def stop_study() -> None:
    """Attempt to stop the study."""
    prolific = st.session_state.get("prolific", {}) or {}
    target_study = prolific.get("study_id")
    api_token = os.environ.get("PROLIFIC_API_TOKEN")

    if not target_study:
        logger.info(
            "No Prolific study id available in session_state; cannot stop study via API."
        )
        return

    if not api_token:
        logger.info(
            "PROLIFIC_API_TOKEN not set; skipping API call to Prolific."
        )
        return

    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json",
    }
    base = os.environ.get("PROLIFIC_API_BASE", "https://api.prolific.com")

    study_actions_url = f"{base}/api/v1/studies/{target_study}"
    payload = {"action": "PAUSE"}
    try:
        resp = requests.post(
            study_actions_url, headers=headers, json=payload, timeout=8
        )
    except Exception as e:
        logger.exception(
            "Error while attempting to call Prolific Studies actions API: %s", e
        )
        return

    if 200 <= resp.status_code < 300:
        logger.info(
            "Successfully sent PAUSE action for study %s via Prolific API",
            target_study,
        )
        return

    logger.warning(
        "Prolific Studies actions API returned %s for study %s; attempting fallback. Response body: %s",
        resp.status_code,
        target_study,
        getattr(resp, "text", ""),
    )

    fallback_target = prolific.get("session_id")
    if not fallback_target:
        return

    fallback_url = f"{base}/api/v1/submissions/{fallback_target}/complete/"
    try:
        fb_resp = requests.post(fallback_url, headers=headers, timeout=8)
    except Exception as e:
        logger.exception("Fallback Prolific request failed: %s", e)
        return

    if 200 <= fb_resp.status_code < 300:
        logger.info(
            "Successfully notified Prolific submission %s", fallback_target
        )
    else:
        logger.warning(
            "Prolific fallback API returned status %s for target %s: %s",
            fb_resp.status_code,
            fallback_target,
            getattr(fb_resp, "text", ""),
        )


def build_questionnaire(page: str, next_page: str = None) -> None:
    """Builds the pre/post questionnaire pages."""
    page_norm = (page or "").lower().strip()

    if page_norm == "screen":
        internal_domains = ["movie"] + st.session_state.get("domains", [])
        domains = [d.replace("_", " ").title() for d in internal_domains]

        options = ["Novice", "Intermediate", "Expert"]

        left, middle, right = st.columns([1, 4, 1])
        with middle:
            with st.form("screen_questionnaire_form", width=800):
                st.markdown("### Self assessment")
                st.write(
                    "Please rate your domain knowledge for each domain using the 3-point scale."
                )

                st.markdown(
                    """
                    <style>
                    div[role="radiogroup"] {
                        display: flex !important;
                        justify-content: space-between !important;
                        gap: 0.4rem !important;
                        width: 600px !important;
                        max-width: 600px !important;
                        min-width: 600px !important;
                    }
                    div[role="radiogroup"] > label {
                        flex: 1 1 0 !important;
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                        min-width: 120px;
                    }
                    div[role="radiogroup"] > label input[type="radio"] {
                        margin-right: 0.45rem !important;
                    }
                    div[role="radiogroup"] > label:first-child {
                        filter: blur(1px) grayscale(60%) !important;
                        opacity: 0.55 !important;
                        pointer-events: none !important;
                        user-select: none !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                placeholder = "Select..."
                options_with_placeholder = [placeholder] + options

                for d in domains:
                    k = f"screen_{d.lower()}"
                    if k not in st.session_state:
                        st.session_state[k] = placeholder

                option_header_cols = st.columns([0.8, 3.2])
                option_header_cols[0].write("\u00a0")
                header_html = "<div style='display:flex; justify-content:space-between; gap:0.4rem; align-items:center; width:600px; max-width:600px; min-width:600px;'>"
                header_html += "<div style='flex: 1 1 0; min-width:120px; display:flex; align-items:center; justify-content:center;'></div>"
                for opt in options:
                    header_html += f"<div style='flex: 1 1 0; min-width:120px; display:flex; align-items:center; justify-content:center; text-align:center; font-size:13px; font-weight:600'>{opt}</div>"
                header_html += "</div>"
                option_header_cols[1].markdown(
                    header_html, unsafe_allow_html=True
                )

                for domain in domains:
                    row_cols = st.columns([1, 4])
                    row_cols[0].write(domain)

                    def _fmt(x):
                        return ""

                    row_cols[1].radio(
                        f"{domain} rating",
                        options_with_placeholder,
                        format_func=_fmt,
                        key=f"screen_{domain.lower()}",
                        horizontal=True,
                        label_visibility="collapsed",
                    )

                btn_l, btn_r = st.columns([6, 1])
                submitted = btn_r.form_submit_button("Continue")

                if submitted:
                    answers = {}
                    missing = []
                    for i, display_domain in enumerate(domains):
                        val = st.session_state.get(
                            f"screen_{display_domain.lower()}", placeholder
                        )
                        if val == placeholder:
                            missing.append(display_domain)
                        internal_domain = internal_domains[i]
                        answers[internal_domain] = (
                            "" if val == placeholder else val
                        )

                    if missing and not st.session_state.get("debug", False):
                        st.error(
                            "Please answer all questions before continuing."
                        )
                    else:
                        st.session_state["screen_answers"] = answers

                        assigned_domain, assigned_expertise = (
                            assign_domain_to_participant(answers)
                        )
                        st.session_state.current_domain = assigned_domain
                        logger.info(
                            f"Assigned participant to domain: {assigned_domain} with expertise: {assigned_expertise}"
                        )

                        save_screen_response(
                            answers,
                            assigned_domain=assigned_domain,
                            assigned_expertise=assigned_expertise,
                        )
                        if "auto_save_conversation" in st.session_state:
                            st.session_state.auto_save_conversation()

                        st.session_state.current_page = "pre"
                        st.rerun()

        return

    if page_norm == "pre":
        if "current_domain" not in st.session_state:
            st.session_state.current_domain = "bicycle"
            logger.warning("current_domain not set, defaulting to bicycle")

        domain_display = st.session_state.current_domain.replace(
            "_", " "
        ).title()

        st.header(f"Knowledge questionnaire -- {domain_display}")
        st.info(
            f"**Domain:** You will be working in the **{domain_display}** domain for this study."
        )
        questions_path = os.path.join(
            f"data/questionnaires/{st.session_state.current_domain}_questions.txt"
        )
        with open(questions_path, "r", encoding="utf-8") as f:
            raw = f.read()

        sections = []
        cur_title = None
        cur_items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                if cur_title and cur_items:
                    sections.append((cur_title, cur_items))
                cur_title = None
                cur_items = []
                continue
            if cur_title is None:
                cur_title = line
                cur_items = []
            else:
                cur_items.append(line)
        if cur_title and cur_items:
            sections.append((cur_title, cur_items))

        options = [
            "Definitely True",
            "Probably True",
            "I don't know",
            "Probably False",
            "Definitely False",
        ]
        placeholder = "Select..."
        options_with_placeholder = [placeholder] + options

        left_col, mid_col, right_col = st.columns([1, 6, 1])
        with mid_col, st.form("knowledge_questionnaire_form", width=1000):
            st.markdown(
                "<div style='font-size:1.05rem; font-weight:600'>Please indicate your agreement with the following statements.</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <style>
                /* Distribute radio labels evenly and keep them centered. */
                div[role="radiogroup"] {
                    display: flex !important;
                    justify-content: space-between !important;
                    gap: 0.4rem !important;
                    align-items: center !important;
                    box-sizing: border-box !important;
                    background: #f7f7f7 !important;
                    min-height: 112px !important; /* increased baseline height */
                    width: 600px !important; /* fixed width to prevent resizing */
                    max-width: 600px !important;
                    min-width: 600px !important;
                    padding: 12px 6px !important; /* balanced vertical padding */
                    border-radius: 0 6px 6px 0 !important;
                    margin-left: -6px !important;
                    margin-bottom: 8px !important;
                    position: relative !important;
                    z-index: 3 !important; /* ensure radios are above other content */
                    box-sizing: border-box !important;
                    overflow: visible !important;
                }
                div[role="radiogroup"] > label {
                    flex: 1 1 0 !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    min-width: 70px;
                    height: auto !important; /* stretch with the radiogroup */
                    padding: 6px 0 !important; /* allow vertical breathing */
                    line-height: 1.1 !important;
                }
                div[role="radiogroup"] > label input[type="radio"] {
                    margin-right: 0.45rem !important;
                }
                /* Style the placeholder (first label) to look disabled and not be interactable */
                div[role="radiogroup"] > label:first-child {
                    filter: blur(1px) grayscale(60%) !important;
                    opacity: 0.55 !important;
                    pointer-events: none !important; /* prevent selection */
                    user-select: none !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            hdr_html = "<div style='display:flex; justify-content:space-between; gap:0.4rem; align-items:center; width:600px; max-width:600px; min-width:600px;'>"
            hdr_html += "<div style='flex: 1 1 0; min-width:70px; display:flex; align-items:center; justify-content:center;'></div>"
            label_breaks = {
                "Definitely True": "Definitely<br>True",
                "Probably True": "Probably<br>True",
                "I don't know": "I don't<br>know",
                "Probably False": "Probably<br>False",
                "Definitely False": "Definitely<br>False",
            }
            for opt in options:
                formatted_opt = label_breaks.get(opt, opt)
                hdr_html += f"<div style='flex: 1 1 0; min-width:70px; display:flex; align-items:center; justify-content:center; margin-left: -20px; text-align:center; font-size:14px; font-weight:600; line-height:1.3;'>{formatted_opt}</div>"
            hdr_html += "</div>"
            answers = {}
            for sec_title, items in sections:
                title_col, header_col = st.columns([3, 5])
                title_col.markdown(
                    f"<h4 style='margin:0; font-size:1.1rem;'>{sec_title}</h4>",
                    unsafe_allow_html=True,
                )
                header_col.markdown(hdr_html, unsafe_allow_html=True)
                for i, stmt in enumerate(items):
                    key = f"q_{sec_title}_{i}".replace(" ", "_")
                    if key not in st.session_state:
                        st.session_state[key] = placeholder

                    left_col, right_col = st.columns([3, 5])

                    left_col.markdown(
                        f"<div style='background:#f7f7f7; padding:12px 14px; "
                        "box-sizing:border-box; border-radius:6px 0 0 6px; "
                        "min-height:112px; display:flex; align-items:center; "
                        "margin-bottom:8px; margin-right:-6px;'>"
                        f"<div style='font-size:1.05rem'>{stmt}</div></div>",
                        unsafe_allow_html=True,
                    )

                    def _fmt(x):
                        return "" if x == placeholder else "\u00a0"

                    answers[key] = right_col.radio(
                        f"Statement {i+1}",
                        options_with_placeholder,
                        format_func=_fmt,
                        key=key,
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                st.markdown(
                    "<div style='height:12px'></div>", unsafe_allow_html=True
                )

            btn_l, btn_r = st.columns([3, 1])
            with btn_l:
                st.write("\u00a0")
            with btn_r:
                submitted = st.form_submit_button("Submit Answers")
            if submitted:
                missing = []
                for sec_title, items in sections:
                    for i, stmt in enumerate(items):
                        key = f"q_{sec_title}_{i}".replace(" ", "_")
                        if st.session_state.get(key) == placeholder:
                            missing.append(stmt)

                if missing and not st.session_state.get("debug", False):
                    st.error("Please answer all statements before submitting.")
                else:
                    normalized = {}
                    for sec_title, items in sections:
                        for i, stmt in enumerate(items):
                            key = f"q_{sec_title}_{i}".replace(" ", "_")
                            normalized[stmt] = st.session_state.get(key)

                    st.session_state["pre_answers"] = answers

                    if hasattr(st.session_state, "auto_save_conversation"):
                        st.session_state.auto_save_conversation()

                    st.session_state.current_page = "start"
                    st.rerun()

    if page_norm == "post":
        questions = load_questionnaire(QUESTIONNAIRE_DIR, page_norm)
        with st.form("post_questionnaire_form"):
            answers = {}
            st.markdown(
                """
                <style>
                .question-block { margin-bottom: 14px; }
                .question-number { font-weight: 600; margin-right: 6px; }
                </style>
                """,
                unsafe_allow_html=True,
            )

            for i, q in enumerate(questions):
                base_text = q.get("question", f"Question {i+1}")
                question_text = f"**{i+1}. {base_text}**"
                question_type = q.get("type", "text")
                key = f"post_q{i+1}"

                st.markdown(
                    f'<div class="question-block">', unsafe_allow_html=True
                )

                if question_type == "scale":
                    scale_min = q.get("scale_min", 1)
                    scale_max = q.get("scale_max", 5)
                    answers[f"q{i+1}"] = st.slider(
                        question_text,
                        min_value=scale_min,
                        max_value=scale_max,
                        key=key,
                        value=(scale_min + scale_max) // 2,
                    )
                elif question_type == "radio":
                    opts = q.get("options", ["Yes", "No"])
                    answers[f"q{i+1}"] = st.radio(
                        question_text,
                        opts,
                        key=key,
                    )
                else:
                    answers[f"q{i+1}"] = st.text_input(question_text, key=key)

                st.markdown("</div>", unsafe_allow_html=True)

            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state["post_answers"] = answers
                if "auto_save_conversation" in st.session_state:
                    st.session_state.auto_save_conversation()
                st.success("Thank you for completing the post-questionnaire!")
                if next_page:
                    st.session_state.current_page = next_page
                    st.rerun()

    else:
        if page and page_norm not in ("screen", "pre", "post"):
            st.info("No questionnaire defined for this page.")
