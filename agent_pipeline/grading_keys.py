"""RSA keypair management for encrypted session logs.

Run generate_keypair() once, as the professor or TA, to produce a private
and a public key. Keep the private key off the repo, on the grader's own
machine only; it's the only thing that can decrypt a student's uploaded
log. The public key is safe to commit to the repo, at
agent_pipeline/keys/grading_public_key.pem. Every student's notebook
uses it to encrypt their own log as it's written. Without the private
key, a student can encrypt but never decrypt or verify their own log,
which is the point: the uploaded file is what they submit, not something
they can inspect or hand-edit afterward.

decrypt_log() is the grading-side counterpart. Run it locally with the
private key against a student's uploaded .jsonl.enc file.
"""

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DEFAULT_PUBLIC_KEY_PATH = "agent_pipeline/keys/grading_public_key.pem"

OAEP_PADDING = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)


def generate_keypair(private_key_path, public_key_path, passphrase: bytes = None):
    """Generates a new RSA-3072 keypair and writes both halves to disk."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    public_key = private_key.public_key()

    encryption = (
        serialization.BestAvailableEncryption(passphrase) if passphrase else serialization.NoEncryption()
    )
    Path(private_key_path).parent.mkdir(parents=True, exist_ok=True)
    with open(private_key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption,
            )
        )
    Path(public_key_path).parent.mkdir(parents=True, exist_ok=True)
    with open(public_key_path, "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
    return private_key_path, public_key_path


def load_public_key(path):
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(f.read())


def load_private_key(path, passphrase: bytes = None):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=passphrase)


def read_student_identity(log_path, private_key_path=None, passphrase: bytes = None) -> dict:
    """Reads a log's header line(s): student_name, student_id, track
    (which notebook produced this log -- "lab2-main",
    "lab2-agent-pipeline"), filename (what this file was actually named
    when it was created), and machine_id (a hashed identifier for the
    computer that ran this session -- see session_logger.py's
    _machine_id() for what it is and, importantly, what it is not).

    **Pass private_key_path to get the identity fields.** As of
    2026-11-01 name, ID and machine_id are encrypted inside the header
    rather than readable in it, because Lab 2 students push their logs
    to public GitHub forks (see the LAB 2 DEVIATION note in
    session_logger.py). Only `track`, `filename`, `created_at` and
    `header_count` come back without a key; the identity fields come
    back None, and `identity_readable` is False to say so rather than
    leaving a caller to read None as "the student left it blank".
    `identity_changed` is None in that case for the same reason.

    Logs in the old plain-text format still read correctly with or
    without a key, so a mixed folder of Lab 1 and Lab 2 submissions
    needs no special handling: `identity_encrypted` reports which format
    each file turned out to be.

    session_logger.py appends a fresh session_key header every time the
    launch-chat cell runs (a new day, a kernel restart, recovering from
    an error mid-session), not just once, so one file can genuinely
    hold several headers -- and nothing stops a student from typing a
    different name or ID into a later one, by typo or otherwise. This
    walks every header in the file, not just the first: the top-level
    student_name/student_id/track/filename/machine_id fields reflect
    the LAST header (a later correction should win over an earlier
    typo), while machine_ids/student_ids_seen/student_names_seen list
    every distinct non-None value seen across ALL headers, and
    identity_changed flags whether student_name or student_id actually
    differed anywhere in the file. Any field can be None for a log
    written before it existed, or if the notebook that created it
    didn't pass one; don't assume every field is populated."""
    headers = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["type"] != "session_key":
                if not headers:
                    raise ValueError(f"{log_path}: expected a session_key header on the first line.")
                continue
            headers.append(entry)

    if not headers:
        return {
            "student_name": None, "student_id": None, "track": None,
            "filename": None, "machine_id": None, "machine_ids": [],
            "student_ids_seen": [], "student_names_seen": [],
            "identity_changed": False, "header_count": 0,
            "identity_encrypted": False, "identity_readable": True,
        }

    # One identity block per header, opened where we can. Each header
    # carries its own wrapped session key, so a file written across
    # several runs needs one decrypt per header, not one for the file.
    private_key = load_private_key(private_key_path, passphrase=passphrase) if private_key_path else None
    resolved = []
    sealed_count = 0
    for h in headers:
        fields = {k: h.get(k) for k in ("student_name", "student_id", "machine_id")}
        sealed = h.get("identity")
        if sealed:
            sealed_count += 1
            if private_key is not None:
                session_key = private_key.decrypt(base64.b64decode(h["encrypted_key"]), OAEP_PADDING)
                plaintext = AESGCM(session_key).decrypt(
                    base64.b64decode(sealed["nonce"]), base64.b64decode(sealed["ciphertext"]), None
                )
                fields.update(json.loads(plaintext))
        resolved.append(fields)

    latest, latest_fields = headers[-1], resolved[-1]
    readable = sealed_count == 0 or private_key is not None
    machine_ids = sorted({r.get("machine_id") for r in resolved if r.get("machine_id")})
    student_ids_seen = sorted({r.get("student_id") for r in resolved if r.get("student_id")})
    student_names_seen = sorted({r.get("student_name") for r in resolved if r.get("student_name")})
    return {
        "student_name": latest_fields.get("student_name"),
        "student_id": latest_fields.get("student_id"),
        "track": latest.get("track"),
        "filename": latest.get("filename"),
        "machine_id": latest_fields.get("machine_id"),
        "machine_ids": machine_ids,
        "student_ids_seen": student_ids_seen,
        "student_names_seen": student_names_seen,
        "identity_changed": (
            len(student_ids_seen) > 1 or len(student_names_seen) > 1 if readable else None
        ),
        "header_count": len(headers),
        "identity_encrypted": sealed_count > 0,
        "identity_readable": readable,
    }


def decrypt_log(log_path, private_key_path, passphrase: bytes = None) -> list[dict]:
    """Decrypts a student's uploaded .jsonl.enc log, returning the list of
    plaintext turn records in order. Requires the matching private key."""
    private_key = load_private_key(private_key_path, passphrase=passphrase)
    records = []
    session_key = None

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["type"] == "session_key":
                encrypted_key = base64.b64decode(entry["encrypted_key"])
                session_key = private_key.decrypt(encrypted_key, OAEP_PADDING)
            elif entry["type"] == "record":
                if session_key is None:
                    raise ValueError(f"{log_path}: found a record before any session_key header.")
                aesgcm = AESGCM(session_key)
                nonce = base64.b64decode(entry["nonce"])
                ciphertext = base64.b64decode(entry["ciphertext"])
                plaintext = aesgcm.decrypt(nonce, ciphertext, None)
                records.append(json.loads(plaintext))
            else:
                raise ValueError(f"{log_path}: unknown log entry type '{entry.get('type')}'.")

    return records
