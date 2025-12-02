import time

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage

from crs.agents.chains import get_response_stream


class ImageMessage(AIMessage):
    """AI message with a separate image URL field for clean rendering."""

    def __init__(self, content: str = "", image_url: str = ""):
        super().__init__(content)
        self.image_url = image_url


TIMER_DURATION = 900


def ensure_timer():
    if "chat_end_ts" not in st.session_state:
        start = st.session_state.get("chat_start_time", time.time())
        st.session_state.chat_start_time = start
        st.session_state.chat_end_ts = start + TIMER_DURATION

    if not st.session_state.get("chat_timer_expired", False):
        if time.time() >= st.session_state.chat_end_ts:
            st.session_state.chat_timer_expired = True
    return st.session_state.get("chat_timer_expired", False)


def build_timer() -> bool:
    if st.session_state.get("current_page") != "chat":
        return False

    expired = ensure_timer()
    if expired:
        st.info("⏰ Time's up! Please choose one recommended item.")
        return False

    end_ts_ms = int(st.session_state.chat_end_ts * 1000)

    components.html(
        f"""
    <div id="timer" style="padding:8px;border:1px solid #ddd;border-radius:8px;">
      ⏱️ Time remaining: <span id="mm">--</span>:<span id="ss">--</span>
    </div>
    <script>
      (function(){{
        if (window.__chatTimerStarted) return;
        window.__chatTimerStarted = true;

        const end = {end_ts_ms};
        const mm = document.getElementById('mm');
        const ss = document.getElementById('ss');

        function tick(){{
          const now = Date.now();
          let remain = Math.max(0, Math.floor((end - now)/1000));
          mm.textContent = String(Math.floor(remain/60)).padStart(2,'0');
          ss.textContent = String(remain%60).padStart(2,'0');

          if (remain <= 0){{
            // reload the TOP page exactly once; no iframe black screen
            const url = new URL(window.top.location.href);
            url.searchParams.set('timer_expired','1');
            window.top.location.replace(url.toString());
            return;
          }}
          setTimeout(tick, 1000);
        }}
        tick();
      }})();
    </script>
    """,
        height=60,
        scrolling=False,
    )

    if st.query_params.get("timer_expired") == "1":
        st.session_state.chat_timer_expired = True
        st.info("⏰ Time's up! Please choose one recommended item.")
        return False

    return True


def coalesce_stream(token_iter, flush_ms=100):
    import time

    buff = []
    last = time.time()
    for t in token_iter:
        buff.append(t)
        now = time.time()
        if (now - last) * 1000 >= flush_ms:
            yield "".join(buff)
            buff = []
            last = now
    if buff:
        yield "".join(buff)


def build_chatbot() -> None:
    """Builds the chatbot interface in the left column."""
    st.session_state.setdefault("assistant_busy", False)
    st.session_state.setdefault("pending_user_message", None)

    if not st.session_state.chat_history:
        welcome = AIMessage(
            "I am your personal recommender assistant. What are you looking for today?"
        )
        st.session_state.chat_history.append(welcome)

    messages_container = st.container(height=550)

    with messages_container:
        for entry in st.session_state.chat_history:
            if isinstance(entry, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(entry.content)
            elif isinstance(entry, ImageMessage):
                with st.chat_message("ai"):
                    st.image(entry.image_url, width=300)
                    st.markdown(entry.content)
            elif isinstance(entry, AIMessage):
                with st.chat_message("ai"):
                    st.markdown(entry.content)

    conversation_state = getattr(st.session_state, "conversation_state", None)
    awaiting_confirmation = False
    if conversation_state and hasattr(conversation_state, "turn_state"):
        awaiting_confirmation = conversation_state.turn_state.get(
            "awaiting_confirmation", False
        )

    user_message = None

    if awaiting_confirmation:
        with messages_container:
            col_left, col1, col2, col_right = st.columns([1, 2, 2, 1])
            with col1:
                if st.button("✅ Yes", use_container_width=True):
                    user_message = "Yes, I want this one."
                    if conversation_state and hasattr(
                        conversation_state, "turn_state"
                    ):
                        conversation_state.turn_state[
                            "user_confirmed_selection"
                        ] = True
            with col2:
                if st.button("🤔 Let me think", use_container_width=True):
                    user_message = "Let me think about it."
        text_input = st.chat_input("Or type your response...")
        if text_input and text_input.strip():
            user_message = text_input
    else:
        user_message = st.chat_input("What are you looking for?")

    if user_message is not None and user_message.strip() != "":
        with messages_container, st.chat_message("user"):
            st.markdown(user_message)

        st.session_state.chat_history.append(HumanMessage(user_message))
        st.session_state.chat_log.append(HumanMessage(user_message))

        conversation_state = getattr(
            st.session_state, "conversation_state", None
        )

        if conversation_state and hasattr(conversation_state, "turn_state"):
            if conversation_state.turn_state.get("awaiting_explanation"):
                print(
                    "📝 User provided explanation, transitioning to post questionnaire..."
                )
                print(
                    f"Current page before transition: {st.session_state.get('current_page')}"
                )

                conversation_state.turn_state["awaiting_explanation"] = False

                st.session_state.auto_save_conversation()

                st.session_state.current_page = "post"
                print(
                    f"Current page after setting: {st.session_state.current_page}"
                )
                st.rerun()

            elif conversation_state.turn_state.get("user_confirmed_selection"):
                print("✅ User confirmed selection, asking for explanation...")
                conversation_state.turn_state["user_confirmed_selection"] = (
                    False
                )
                conversation_state.turn_state["awaiting_confirmation"] = False

                _selected_item_name = conversation_state.turn_state.get(
                    "selected_item_name", "this item"
                )
                explanation_prompt = (
                    "Great! Before we finalize, could you briefly explain in a "
                    "sentence or two why you think this option is the best "
                    "choice for you?"
                )

                explanation_msg = AIMessage(explanation_prompt)
                st.session_state.chat_history.append(explanation_msg)
                st.session_state.chat_log.append(explanation_msg)
                st.session_state.auto_save_conversation()

                conversation_state.turn_state["awaiting_explanation"] = True

                st.rerun()

        response_stream = get_response_stream(
            st.session_state.task,
            st.session_state.chat_history,
            st.session_state.get("model_name", "gpt-4.1-mini"),
        )
        image_url = response_stream.get("image_url", "")
        with messages_container, st.chat_message("ai"):
            if image_url:
                st.image(image_url, width=300)
            raw_stream = response_stream["stream"]
            response = st.write_stream(
                coalesce_stream(raw_stream, flush_ms=100)
            )

        if image_url:
            ai_msg = ImageMessage(response or "", image_url=image_url)
        else:
            ai_msg = AIMessage(response or "")

        st.session_state.chat_history.append(ai_msg)
        st.session_state.chat_log.append(ai_msg)
        st.session_state.auto_save_conversation()
