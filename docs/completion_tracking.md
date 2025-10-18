# Completion Tracking Implementation

## Summary

Added completion tracking to identify participants who completed the study vs. those who dropped out after the initial screening.

## Changes Made

### 1. Updated Code (`crs/main.py`)
- Added `"completed"` field to the conversation data saved by `auto_save_conversation()`
- Defaults to `False`, set to `True` when participant reaches the end page

### 2. Updated End Page (`crs/components/pages.py`)
- When a participant reaches the "end" page (study complete):
  - Sets `st.session_state.study_completed = True`
  - Saves the conversation with this flag

### 3. Updated Existing Data

#### Conversation Files (`exports/conversations/*.json`)
- Script: `scripts/add_completed_field.py`
- Added `"completed": true/false` field to all 158 conversation files
- Completion determined by presence of `post_task_answers` (non-empty)
- Results: 87 completed (55.1%), 71 incomplete (44.9%)

#### Screen Results (`exports/screen_results.jsonl`)
- Script: `scripts/update_screen_results.py`
- Added `"completed": true/false` field to all 157 entries
- Matched each screen entry with corresponding conversation file by Prolific ID and session ID
- Results: 79 completed (50.3%), 78 incomplete (49.7%)

## Results

### Completion Statistics

**Conversation Files (all participants who started the task):**
- Total: 158 files
- Completed: 87 (55.1%)
- Dropped out: 71 (44.9%)

**Screen Results (all participants who passed screening):**
- Total: 157 entries
- Completed: 79 (50.3%)
- Dropped out: 78 (49.7%)

### Drop-out Analysis

The difference between screen results (157) and conversation files (158) suggests:
- 1 participant may have had multiple sessions or a restart

The drop-out rate of ~50% indicates that about half of participants who complete the screening questionnaire do not complete the full study.

## File Structure

### Conversation Files Format
```json
{
    "last_saved": "timestamp",
    "prolific": {...},
    "screen_answers": {...},
    "pre_task_answers": {...},
    "post_task_answers": {...},
    "current_domain": "...",
    "task_version": "...",
    "completed": true/false,  // NEW FIELD
    "messages": [...]
}
```

### Screen Results Format
```json
{
    "timestamp": "...",
    "answers": {...},
    "assigned_domain": "...",
    "assigned_expertise": "...",
    "prolific": {...},
    "completed": true/false  // NEW FIELD
}
```

## Future Usage

To count only completed participants:
```python
import json

# For conversation files
completed = sum(1 for f in glob.glob('exports/conversations/*.json') 
                if json.load(open(f)).get('completed'))

# For screen results
with open('exports/screen_results.jsonl', 'r') as f:
    completed = sum(1 for line in f if json.loads(line).get('completed'))
```

## Scripts Created

1. **`scripts/add_completed_field.py`** - Adds completion field to conversation files
2. **`scripts/update_screen_results.py`** - Adds completion field to screen_results.jsonl

Both scripts can be re-run safely on new data as they skip already-updated entries.
