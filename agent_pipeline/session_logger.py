"""Append-only, encrypted JSONL logger for one student session.

Turns are written immediately, not buffered, so a kernel crash or Colab
disconnect doesn't lose grading data. Each turn is encrypted before it
touches disk, using AES-GCM with a per-session key wrapped by the
grader's RSA public key (see grading_keys.py). The notebook only ever
holds the public key, so it can write an encrypted log but can never
read it back. That's the point: the uploaded .jsonl.enc file is the
submission, not something a student can open in a text editor and hand-edit.

The log's job is to record the exact sequence of prompts and tool calls,
name and args, since the professor's grading process re-runs that
sequence independently to verify results. The log's numbers don't need
to be trusted, only the recorded sequence does. Encryption on top of
that means a student can't read their own log back to see what's being
tracked, and can't casually hand-edit individual entries before
uploading.
"""

import base64
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from langchain_core.messages import AIMessage, ToolMessage

from .grading_keys import DEFAULT_PUBLIC_KEY_PATH, OAEP_PADDING, load_public_key
from .message_text import as_text


def _machine_id() -> str:
    """A short, stable-per-machine identifier, for cross-checking that
    submissions claiming to be different students didn't come from the
    same computer -- not a raw MAC address (uuid.getnode()), which is
    real hardware-identifying information with no reason to leave a
    student's machine in the clear. Hashed down to 12 hex characters:
    enough to notice two logs sharing a machine.

    Two caveats, both measured on 2026-11-01 rather than assumed. First,
    12 hex characters is 48 bits of SHA-256 over a 48-bit input, so a
    published hash can be brute-forced back to the address it came from
    (seconds, given a guess at the vendor prefix) -- which is part of why
    this value now travels inside the encrypted part of the file instead
    of the header's plain text. Second, it is NOT reliably stable per
    machine: six headers written from one laptop carried five different
    values, because uuid.getnode() falls back to a fresh random node
    whenever it cannot read a hardware address, which is what happens
    under the notebooks' own environment. So treat a match between two
    submissions as a real signal, and a mismatch as no information at
    all."""
    raw = str(uuid.getnode())
    return hashlib.sha256(raw.encode("ascii")).hexdigest()[:12]


class SessionLogger:
    def __init__(self, log_path, student_name, student_id, track=None, public_key_path=DEFAULT_PUBLIC_KEY_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._turn_index = 0
        # Exposed as an attribute, not just embedded in the header below,
        # so other code that already holds this logger (chat_widget.py,
        # for the saved plots) can tag its own output with the exact same
        # machine_id instead of recomputing it separately.
        self.machine_id = _machine_id()

        public_key = load_public_key(public_key_path)
        session_key = AESGCM.generate_key(bit_length=256)
        self._aesgcm = AESGCM(session_key)
        encrypted_session_key = public_key.encrypt(session_key, OAEP_PADDING)

        # LAB 2 DEVIATION (2026-11-01): who the student is no longer
        # appears in this header in plain text. It used to -- name, ID and
        # machine_id sat here readable, so a batch of submissions could be
        # matched to students before anyone ran a decrypt -- and in Lab 1
        # that was harmless, because nothing asked students to publish the
        # file. Lab 2 does: session_logs/ is deliberately tracked by git,
        # since the encrypted logs ARE the submission, and students work in
        # public forks of a public repository. The result was a real name
        # and a real nine-digit student ID published on the open internet,
        # permanently, for every student who handed in. So the three
        # identifying fields are sealed with this session's own AES key
        # alongside every turn, and read back by read_student_identity()
        # in grading_keys.py, which takes the private key and returns the
        # same fields it always did.
        #
        # What stays readable is deliberately non-identifying: `track`
        # (which notebook wrote this log), `filename` (what the file was
        # called when it was created, so a rename or a flattened ZIP
        # upload cannot lose that -- all three notebooks share one
        # session_logs/ folder), and `created_at`. That is enough to sort
        # a folder of submissions by notebook and to spot an empty or
        # truncated file without a key, which is all the plain text was
        # ever used for on our side.
        header = {
            "type": "session_key",
            "track": track,
            "filename": self.log_path.name,
            "encrypted_key": base64.b64encode(encrypted_session_key).decode("ascii"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identity": self._seal(
                {
                    "student_name": student_name,
                    "student_id": student_id,
                    "machine_id": self.machine_id,
                }
            ),
        }
        # Appended, not truncated: the notebooks explicitly tell students
        # to re-run the launch-chat cell mid-session (e.g. after building
        # a new tool in the dev notebook, or recovering from an error),
        # and each SessionLogger() construction generates its own fresh
        # AES session key -- opening in "w" mode here would silently
        # destroy every turn logged by a previous instance the moment
        # that cell reran, exactly the "doesn't lose grading data"
        # guarantee this file's own docstring promises. decrypt_log()
        # already handles multiple session_key headers in one file
        # correctly (each one just becomes the active key for whatever
        # records follow it), so appending here is enough on its own --
        # nothing on the decrypt side needs to change.
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")

    def _seal(self, obj) -> dict:
        """AES-GCM encrypts one JSON-serialisable object under this
        session's key, in the same {nonce, ciphertext} shape every
        encrypted part of this file uses. Shared by the header's identity
        block and by each logged turn, so there is one encryption path to
        get right rather than two."""
        plaintext = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, None)
        return {
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }

    def _write_encrypted(self, record: dict) -> None:
        line = {"type": "record", **self._seal(record)}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def log_turn(self, student_prompt: str, new_messages: list, ungrounded_numbers=None) -> dict:
        self._turn_index += 1

        tool_call = None
        tool_result_summary = None
        agent_response_text = None
        dropped_tool_calls = []

        for m in new_messages:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                first, *rest = m.tool_calls
                tool_call = {"name": first["name"], "args": first["args"]}
                dropped_tool_calls.extend({"name": t["name"], "args": t["args"]} for t in rest)
            elif isinstance(m, ToolMessage):
                tool_result_summary = m.content
            elif isinstance(m, AIMessage):
                agent_response_text = as_text(m.content)

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "turn_index": self._turn_index,
            "student_prompt": student_prompt,
            "tool_call": tool_call,
            "dropped_tool_calls": dropped_tool_calls,
            "tool_result_summary": tool_result_summary,
            "agent_response_text": agent_response_text,
            "ungrounded_numbers": ungrounded_numbers or [],
        }
        self._write_encrypted(record)
        return record
