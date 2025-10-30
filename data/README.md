# Data

This folder contains the collected dialogues dataset and the data used in support of collecting it.

Overview
- `data/dialogues/` — Collected conversation JSONs with one file per conversation. See details below.
- `data/questionnaires/` — Contains pre-task and post-task questionnaires used to collect user domain knowledge and feedback.
- `data/tasks/` — Contains background stories for which participants had to find the optimal product. For each domain we curated 6 background stories (3 long variants and 3 short).


## Dialogues

This dataset stores one JSON object per conversation. Files are under `data/dialogues`.

The fields in the dialogues are:

- `last_saved` — ISO timestamp when the conversation was last written.
- `domain` — topic or domain of the task (One of `bicycle`, `digital camera`, `laptop`, `smartwatch`, `running shoes`).
- `target` — object describing the target item (contains `id` and `title`).
- `task_version` — task variant identifier (`long` or `short`).
- `completed` — boolean flag indicating whether the participant completed the task.
- `self_assessment` — per-domain self-assessment values.
- `pre_task_answers` — contains survey responses collected before the task.
- `post_task_answers` — contains post-task survey fields.
- `recommendations` — array of recommended item ids produced during the conversation.
- `messages` — ordered message transcript (see `messages` section below).

### messages

An ordered array of message objects. Each message contains:
  * `role` (string): one of `human`, `ai`, or `system`
    - `human`: participant utterances (what the human said).
    - `ai`: assistant/model responses.
    - `system`: internal/system messages not visible to the user (for example parsed preferences, retrieved items, dialogue flow decisions).
  * `content` (string): the textual content of the message.

