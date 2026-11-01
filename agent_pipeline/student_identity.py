"""Where a student's name and ID come from (LAB 2 ADDITION, 2026-11-01).

Both used to be typed straight into a notebook cell: `STUDENT_NAME =
"..."` in the Master's Section 8, and a pair of text boxes in the
AgentPipeline's setup widget. Both of those end up published, because
students work in public forks of a public repository and are told to
commit their notebooks with outputs visible -- the ID sat in the cell
source, in the cell's printed output, and in the session log's header.

So identity lives in `config/.env` instead: the one file in the
repository that is git-ignored, that students already create by copying
`config/.env.example`, and that they are already told to keep private
because their API key is in it. Nothing identifying goes into a notebook
cell, an output, or a log header in plain text.

This reads that file directly rather than going through os.environ, on
purpose. `load_dotenv()` does not overwrite a variable that is already
set, so a student who starts a kernel with a typo in their ID, fixes the
file, and re-runs the cell would otherwise keep writing the typo for the
rest of the session -- and they cannot read their own log back to notice.
"""

from pathlib import Path

# Values from .env.example, plus the obvious near-misses. A log written
# under any of these cannot be credited to anybody, so they are treated
# as "not filled in" rather than as a name.
_PLACEHOLDERS = {
    "",
    "your name",
    "your name here",
    "your full name",
    "your student id",
    "your-name",
    "your-student-id",
    "student",
    "s000000000",
    "000000000",
    "123456789",
    "dm2026 ta",
}

_MISSING_FILE = (
    "config/.env not found. Copy config/.env.example to config/.env, then fill in "
    "STUDENT_NAME, STUDENT_ID and your API key. The file is git-ignored, so what you "
    "put in it stays on your own machine -- that is the point of keeping it there."
)

_NOT_FILLED = (
    "Put your own name and student ID in config/.env:\n"
    '    STUDENT_NAME = "Your Name"\n'
    '    STUDENT_ID   = "113000000"\n'
    "then run this cell again. Your work is graded under that identity, and a log "
    "written under the placeholder cannot be credited to anybody.\n"
    "The file is git-ignored: your ID is recorded in your session log, encrypted, "
    "and never appears in a notebook, an output, or anything you push."
)


def load_student_identity(env_path="config/.env") -> tuple[str, str]:
    """Returns (student_name, student_id) read from `config/.env`.

    Raises ValueError with instructions if the file is missing or either
    value is still a placeholder -- loudly, rather than letting a whole
    session be logged under a name that cannot be graded.
    """
    path = Path(env_path)
    if not path.exists():
        raise ValueError(_MISSING_FILE)

    from dotenv import dotenv_values

    values = dotenv_values(path)
    name = (values.get("STUDENT_NAME") or "").strip()
    student_id = (values.get("STUDENT_ID") or "").strip()

    if name.lower() in _PLACEHOLDERS or student_id.lower() in _PLACEHOLDERS:
        raise ValueError(_NOT_FILLED)

    return name, student_id


def describe_student_identity(env_path="config/.env") -> str:
    """A one-line confirmation to print in a notebook: enough for a
    student to catch a typo, little enough that the line is safe to
    commit in a cell's output. The ID is shown as its first two and last
    two characters, which is what a typo usually breaks."""
    name, student_id = load_student_identity(env_path)
    masked = student_id if len(student_id) < 5 else f"{student_id[:2]}{'*' * (len(student_id) - 4)}{student_id[-2:]}"
    initials = " ".join(part[0].upper() + "." for part in name.split() if part)
    return f"Identity loaded from {env_path}: {initials} / {masked} ({len(student_id)} characters)"
