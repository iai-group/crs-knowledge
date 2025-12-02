import streamlit as st


def build_introduction() -> None:
    """Builds the introduction page."""
    st.title("The Recommendation Game")

    st.write(
        "In this game, you'll interact with an AI assistant to find a hidden **target item** from a collection. "
        "You'll get a short background story describing what the ideal item should handle or achieve."
    )

    st.markdown(
        "**Your goal:** figure out which recommended item best fits the scenario."
    )

    st.write(
        "During the conversation, the assistant will suggest several options, including the target and some decoys. "
        "Use the scenario to ask smart questions, compare features, and test which one truly fits."
    )

    st.markdown("### Tips")
    st.markdown(
        "- Use natural language; there's no right or wrong way to write.\n"
        "- Ask about attributes or features if you're unsure what matters for the scenario.\n"
        "- Think carefully about differences between items: why does one fit the background better than the others?\n"
        "- When explaining your choice, mention specific features or technical details.\n"
    )
