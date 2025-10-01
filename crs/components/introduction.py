import streamlit as st


def build_introduction() -> None:
    """Builds the introduction page."""
    st.title("The Recommendation Game")

    st.write(
        "In this task, you will interact with an AI assistant to receive a recommendation for a specific item. "
        "You'll be given a short background scenario and an image. "
        "The image and the scenario both refer to a specific **target** item from a collection the assistant knows.",
    )

    st.markdown(
        "**Your goal is to guide the conversation in a way that leads the assistant to recommend that specific target item.** "
    )

    st.write(
        "You may receive recommendations that aren't quite right. Use those moments to clarify what you really need, and explain how the target item differs. "
        "The more you say, the more accurate the recommendation can be."
    )

    st.markdown("### Tips")
    st.markdown(
        "- Use natural language; there’s no right or wrong way to write.\n"
        "- Ask questions about attributes if you're unsure what they are or what is available.\n"
        "- Think of this like a puzzle: you're helping the assistant figure out what you're aiming for."
    )
