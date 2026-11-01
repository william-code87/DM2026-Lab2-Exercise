"""Notebook-embedded backend picker.

This replaces an earlier version that used input() for this step. input()
depends on the notebook frontend having its stdin channel wired up to the
kernel, and not every frontend does that by default -- some just sit at
"kernel busy" forever with no visible box to type into and no error, which
looks exactly like the notebook has frozen. A clickable widget doesn't have
that dependency: it renders as normal cell output the same way the chat
widget below it already does.
"""

import ipywidgets as widgets
from IPython.display import display

from .llm_backends import get_api_keys
from .student_identity import describe_student_identity, load_student_identity


def select_backend(valid_backends=("groq", "gemini")):
    """Displays a backend dropdown, a fallback checkbox, and a Confirm
    button. Returns a plain dict, empty until the student picks a backend
    and clicks Confirm, at which point it's filled in with
    "student_name", "student_id", "backend", and "multi_provider". Run
    this cell, click Confirm, then move on -- the next cell checks for
    those keys and tells you plainly if you skipped this step.

    LAB 2 DEVIATION (2026-11-01): the name and student ID text boxes are
    gone. Whatever a student typed into them was echoed into this cell's
    output, which they then committed to a public fork, and ipywidgets
    persists a text box's value into the notebook's own metadata when
    JupyterLab saves widget state -- so the ID could survive even
    clearing the outputs. Both values are read from the git-ignored
    config/.env now (see student_identity.py); the dict keys are
    unchanged, so the launch cell below needs no edit.

    This only checks that a usable key exists; it doesn't read or store
    one. The graph-launch cell builds the actual model itself, straight
    from config/.env, through build_llm_for_backend -- which is also
    where a backend with several keys configured (GOOGLE_API_KEY_1
    through GOOGLE_API_KEY_10, say) gets turned into a model that
    rotates between all of them. See llm_backends.py.

    Both identifiers are encrypted into the session log alongside every
    turn (see session_logger.py), so a submission is matched to a student
    with the grading key rather than by reading the file."""
    choice = {}

    backend_dropdown = widgets.Dropdown(options=valid_backends, description="Backend:")
    fallback_checkbox = widgets.Checkbox(value=False, description="Enable multi-provider fallback")
    confirm_button = widgets.Button(description="Confirm", button_style="primary")
    status_output = widgets.Output()

    def on_confirm(_=None):
        status_output.clear_output()
        backend = backend_dropdown.value
        with status_output:
            try:
                student_name, student_id = load_student_identity()
            except ValueError as exc:
                print(f"Error: {exc}")
                return
            keys = get_api_keys(backend)
            if not keys:
                print(
                    f"Error: no usable key found for '{backend}'. Copy config/.env.example to "
                    f"config/.env and paste in your key, then run this cell again."
                )
                return
            choice["student_name"] = student_name
            choice["student_id"] = student_id
            choice["backend"] = backend
            choice["multi_provider"] = fallback_checkbox.value
            # Masked, not echoed: this line is saved into the notebook
            # as cell output and pushed to a public repository.
            print(describe_student_identity())
            key_word = "key" if len(keys) == 1 else "keys"
            print(f"Using backend: {backend}. {len(keys)} API {key_word} loaded.")
            print(f"Multi-provider fallback: {'enabled' if fallback_checkbox.value else 'disabled'}.")
            print("Set. Move on to the next cell.")

    confirm_button.on_click(on_confirm)
    display(widgets.VBox([backend_dropdown, fallback_checkbox, confirm_button, status_output]))
    return choice
