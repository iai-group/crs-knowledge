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
    # Prefer loading precomputed counts if available.
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

    # Fallback: count from the raw JSONL screen results (original behaviour)
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

                # Get the assigned domain and expertise from the record
                assigned_domain = rec.get("assigned_domain", None)
                assigned_expertise = rec.get("assigned_expertise", None)

                if assigned_domain and assigned_expertise:
                    # Create key like "bicycle-expert"
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
    # Get current counts of assigned domain-expertise combinations
    counts = count_participants_by_domain_expertise(out_path)

    # Get available domains from session state (those with data files)
    available_domains = st.session_state.get("domains", [])

    # Define expertise ranking (lower number = higher priority)
    expertise_rank = {
        "expert": 0,
        "intermediate": 2,
        "novice": 1,
    }

    # Build list of candidates: (count, expertise_rank, domain_name, expertise_level)
    candidates = []

    for domain_key, knowledge_level in participant_answers.items():
        if not knowledge_level or knowledge_level == "":
            continue

        # domain_key is already in internal format (e.g., "bicycle", "digital_camera")
        # Only consider domains that are actually available (have data files)
        if domain_key not in available_domains:
            continue

        # Get the count for this domain-expertise combination
        count_key = f"{domain_key}-{knowledge_level.lower()}"
        count = counts.get(count_key, 0)

        # Get the expertise rank (lower is better)
        expertise_priority = expertise_rank.get(knowledge_level.lower(), 999)

        # Add to candidates: (count, expertise_rank, domain_name, expertise_level)
        # Sort will prioritize lowest count, then lowest rank (highest expertise)
        candidates.append(
            (count, expertise_priority, domain_key, knowledge_level)
        )

    if not candidates:
        # Fallback to bicycle-novice if no valid answers for available domains
        logger.warning(
            "No valid domain answers found for available domains, defaulting to bicycle-Novice"
        )
        return ("bicycle", "Novice")

    # Sort by count (ascending), then by expertise rank (ascending = Expert first)
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
    # Only use the study identifier here — do NOT use the user id.
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

    # Use the Studies actions endpoint to request a PAUSE (per public docs).
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

    # If the action worked, we're done. Otherwise attempt fallback.
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

    # Fallback: try to mark a submission as complete using session_id (not user id)
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
    # Normalize page input (avoid casing/whitespace mismatches)
    page_norm = (page or "").lower().strip()
    # st.header(f"{page_norm.capitalize()} Questionnaire")

    # ----------- Front Questionnaire (Grid-style self-assessment) -----------
    if page_norm == "screen":
        # Get domains from session state and transform for display
        # Add "Movies" as it's used for screening but not in the study
        internal_domains = ["movie"] + st.session_state.get("domains", [])
        # Transform to display format: remove underscores and capitalize each word
        domains = [d.replace("_", " ").title() for d in internal_domains]

        options = ["Novice", "Intermediate", "Expert"]

        # Center the questionnaire by creating a three-column layout and
        # placing the form in the middle (narrower) column.
        left, middle, right = st.columns([1, 4, 1])
        with middle:
            with st.form("screen_questionnaire_form", width=800):
                st.markdown("### Self assessment")
                st.write(
                    "Please rate your domain knowledge for each domain using the 3-point scale."
                )

                # Inject CSS to make Streamlit horizontal radio groups distribute
                # their options evenly and align with the header grid above.
                # This targets the radiogroup container and its labels. It's
                # a best-effort selector and may be fragile across Streamlit
                # versions, but generally works for horizontal radios.
                st.markdown(
                    """
                    <style>
                    /* Ensure radio groups lay out options evenly */
                    div[role="radiogroup"] {
                        display: flex !important;
                        justify-content: space-between !important;
                        gap: 0.4rem !important;
                        width: 600px !important; /* fixed width to prevent resizing */
                        max-width: 600px !important;
                        min-width: 600px !important;
                    }
                    /* Make each label take equal space and center its contents */
                    div[role="radiogroup"] > label {
                        flex: 1 1 0 !important;
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                        min-width: 120px;
                    }
                    /* Slight spacing between radio and its label text */
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

                # Use a placeholder option so nothing is preselected, but render as radios
                placeholder = "Select..."
                options_with_placeholder = [placeholder] + options

                # Ensure session_state keys exist so the widgets have a stable key/value
                for d in domains:
                    k = f"screen_{d.lower()}"
                    if k not in st.session_state:
                        st.session_state[k] = placeholder

                # Render a compact horizontal radio per domain (starts on placeholder)
                # We want the option titles above the grid, so render a small
                # header row with the titles and then place the radios below.
                # Make the label column narrower so labels sit closer to controls
                option_header_cols = st.columns([0.8, 3.2])
                # left column reserved for the domain label (empty header)
                option_header_cols[0].write("\u00a0")  # non-breaking space
                # create a row in the right column to show each option title
                # Use flexbox to match radio button spacing behavior
                header_html = "<div style='display:flex; justify-content:space-between; gap:0.4rem; align-items:center; width:600px; max-width:600px; min-width:600px;'>"
                # empty cell to align with the placeholder radio
                header_html += "<div style='flex: 1 1 0; min-width:120px; display:flex; align-items:center; justify-content:center;'></div>"
                for opt in options:
                    header_html += f"<div style='flex: 1 1 0; min-width:120px; display:flex; align-items:center; justify-content:center; text-align:center; font-size:13px; font-weight:600'>{opt}</div>"
                header_html += "</div>"
                option_header_cols[1].markdown(
                    header_html, unsafe_allow_html=True
                )

                for domain in domains:
                    # left label column and right control column inside the centered column
                    row_cols = st.columns([1, 4])
                    # use write() instead of bold markdown to avoid extra vertical padding
                    row_cols[0].write(domain)

                    # horizontal radios render like a row of choices (compact/grid-like)
                    # Accessibility: avoid passing an empty label. We visually
                    # render the domain name in the left column and provide a
                    # non-empty label for the radio control which is hidden
                    # from sighted users via `label_visibility="hidden"`.
                    # Hide the placeholder label, and return a single
                    # non-breaking space for real options so the radio
                    # buttons have a minimal uniform label width. The
                    # grid header above handles the visible titles and
                    # aligns with the radio columns.
                    def _fmt(x):
                        return ""  # if x == options[2] else "\u00a0" * 40

                    row_cols[1].radio(
                        f"{domain} rating",
                        options_with_placeholder,
                        format_func=_fmt,
                        key=f"screen_{domain.lower()}",
                        horizontal=True,
                        label_visibility="collapsed",
                    )

                # Always show the submit button; validate when submitted
                btn_l, btn_r = st.columns([6, 1])
                submitted = btn_r.form_submit_button("Continue")

                if submitted:
                    # Read current values from session_state and validate
                    # Store answers using internal domain names (not display names)
                    answers = {}
                    missing = []
                    for i, display_domain in enumerate(domains):
                        val = st.session_state.get(
                            f"screen_{display_domain.lower()}", placeholder
                        )
                        if val == placeholder:
                            missing.append(display_domain)
                        # Use internal domain name as key (from internal_domains list)
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

                        # Assign domain to participant based on balancing
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
                        # Auto-save if available
                        if "auto_save_conversation" in st.session_state:
                            st.session_state.auto_save_conversation()
                        # Move to the next page if provided

                        # count = count_bicycle_category(
                        #     answers.get("bicycle", "")
                        # )
                        # if (
                        #     count > MAX_GROUP_PARTICIPANTS
                        #     and not st.session_state.get("debug", False)
                        # ):
                        #     st.session_state.current_page = "prolific_redirect"
                        # else:
                        #     count_all_categories = sum(
                        #         count_bicycle_category(opt)
                        #         for opt in ["Novice", "Intermediate", "Expert"]
                        #     )
                        #     if (
                        #         count_all_categories
                        #         > MAX_GROUP_PARTICIPANTS * 3
                        #         and not st.session_state.get("debug", False)
                        #     ):
                        #         stop_study()
                        st.session_state.current_page = "pre"
                        st.rerun()

        return

    if page_norm == "pre":
        # Use the domain assigned during screen questionnaire
        # If not set (e.g., debug mode), default to "bicycle"
        if "current_domain" not in st.session_state:
            st.session_state.current_domain = "bicycle"
            logger.warning("current_domain not set, defaulting to bicycle")

        # Format domain name for display (e.g., "digital_camera" → "Digital Camera")
        domain_display = st.session_state.current_domain.replace(
            "_", " "
        ).title()

        # Reuse the original knowledge questionnaire rendering logic from main
        st.header(f"Knowledge questionnaire -- {domain_display}")
        st.info(
            f"**Domain:** You will be working in the **{domain_display}** domain for this study."
        )
        # current_domain is already lowercase, so use it directly
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
            hdr_html += "<div style='flex: 1 1 0; min-width:70px; display:flex; align-items:center; justify-content:center;'></div>"  # placeholder column
            # Break labels into two lines more naturally
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

    # ----------- Post Questionnaire (loaded from JSON) -----------
    if page_norm == "post":
        questions = load_questionnaire(QUESTIONNAIRE_DIR, page_norm)
        with st.form("post_questionnaire_form"):
            answers = {}
            # small CSS spacer class for consistent vertical spacing between questions
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
                # Prepend numbering to the visible question text so users can
                # quickly see progress; keep the original text in case the
                # JSON already includes numbering.
                base_text = q.get("question", f"Question {i+1}")
                question_text = f"**{i+1}. {base_text}**"
                question_type = q.get("type", "text")
                key = f"post_q{i+1}"

                # Wrap each question in a div so our CSS spacing applies
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
        # Only show a message if a non-empty unknown page name was supplied
        if page and page_norm not in ("screen", "pre", "post"):
            st.info("No questionnaire defined for this page.")
