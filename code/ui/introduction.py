import streamlit as st


def build_introduction() -> None:
    """Builds the introduction page."""
    st.title("Recommendation Game")

    st.write(
        "In this task, you will interact with an AI assistant to receive a recommendation for a specific item. "
        "You'll start by selecting a domain and will be given a short background scenario and an image. "
        "The image and the scenario both refer to a specific target item from a collection the AI knows."
    )

    st.write(
        "Your goal is to guide the conversation in a way that leads the AI to recommend that specific target item—or the closest match. "
        "You can use your knowledge of the domain, describe your preferences in your own words, or ask questions to better understand the available options."
    )

    st.write(
        "The AI cannot see the image you've been given, and it won't know which item you're referring to unless you describe what you're looking for. "
        "It can answer questions about the domain, explain features, and make educated guesses based on your input."
    )

    st.write(
        "You may receive recommendations that aren't quite right. Use those moments to clarify what you really want, and explain how your intended item differs. "
        "The more you say, the more accurate the recommendation can be."
    )

    st.markdown("### Tips")
    st.markdown(
        "- Use natural language--there’s no right or wrong way to speak."
    )
    st.markdown(
        "- Describe what you're looking for in terms of function, vibe, or features."
    )
    st.markdown(
        "- Ask questions about attributes if you're unsure what's available."
    )
    st.markdown(
        "- Think of this like a puzzle: you're helping the AI figure out what you're aiming for."
    )
