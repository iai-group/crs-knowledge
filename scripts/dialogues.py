"""This script is for getting clean dialogues from the log data."""

import json
import os


def load_exported_conversation(file_path: str) -> dict:
    """
    Load the exported conversation from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing the conversation.

    Returns:
        dict: The loaded conversation data.
    """
    with open(file_path, "r") as file:
        conversation = json.load(file)
    return conversation


def clean_dialogues(conversation: dict) -> list:
    """
    Clean the dialogues from the conversation data.

    Args:
        conversation (dict): The conversation data.

    Returns:
        list: A list of cleaned dialogues.
    """
    cleaned_dialogues = []
    for entry in conversation["messages"]:
        if entry["role"] == "human":
            cleaned_dialogues.append(
                {"role": "user", "content": entry["content"]}
            )
        elif entry["role"] == "ai":
            cleaned_dialogues.append(
                {"role": "agent", "content": entry["content"]}
            )
    conversation["messages"] = cleaned_dialogues
    return conversation


if __name__ == "__main__":
    # Example usage
    conversations_folder = "exports/conversations"
    output_folder = "exports/cleaned_conversations"
    archove_folder = "exports/archived_conversations"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(archove_folder, exist_ok=True)
    # Move the files to the archive folder

    for filename in os.listdir(conversations_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(conversations_folder, filename)
            conversation = load_exported_conversation(file_path)
            cleaned_dialogues = clean_dialogues(conversation)
            output_file_path = os.path.join(output_folder, filename)
            with open(output_file_path, "w") as output_file:
                json.dump(cleaned_dialogues, output_file, indent=4)
            print(f"Cleaned dialogues saved to {output_file_path}")
            # Move the original file to the archive folder
            archive_file_path = os.path.join(archove_folder, filename)
            os.rename(file_path, archive_file_path)
            print(f"Archived original file to {archive_file_path}")
