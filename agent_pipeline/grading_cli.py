"""Professor/TA-only command-line tool for the encrypted session logs.
It's never imported by student-facing code (tools_*.py, graph.py,
chat_widget.py); run it directly, outside the student notebook flow.

Generate a keypair once, before distributing the repo to students:
    python -m agent_pipeline.grading_cli generate-keys

Commit only the printed public key path to the repo. Keep the private key
path off the repo entirely (e.g. outside the working directory), on
whatever machine will grade submissions.

The student's name and ID are encrypted inside the log as of 2026-11-01
(Lab 2 students push their logs to public forks), so both decrypt
commands below need --private-key to report who a file belongs to, not
just to read its turns. Logs in the older plain-text format still work.

Decrypt a student's uploaded log for grading:
    python -m agent_pipeline.grading_cli decrypt path/to/student_session.jsonl.enc \\
        --private-key /path/to/grading_private_key.pem

Decrypt a whole folder of submissions at once, one per student:
    python -m agent_pipeline.grading_cli decrypt-batch path/to/submissions/ \\
        --private-key /path/to/grading_private_key.pem --output-dir path/to/decrypted/
"""

import argparse
import json
import sys
from pathlib import Path

from .grading_keys import DEFAULT_PUBLIC_KEY_PATH, decrypt_log, generate_keypair, read_student_identity


def _cmd_generate_keys(args):
    priv, pub = generate_keypair(args.private_key, args.public_key)
    print(f"Private key written to: {priv}  -- KEEP THIS OFF THE REPO, off any shared drive.")
    print(f"Public key written to:  {pub}  -- safe to commit; students' notebooks load this.")


def _cmd_decrypt(args):
    identity = read_student_identity(args.log_path, args.private_key)
    student_name, student_id, track = identity["student_name"], identity["student_id"], identity["track"]
    if student_name or student_id:
        track_suffix = f" [{track}]" if track else ""
        print(f"Student: {student_name} ({student_id}){track_suffix}")
        if identity["filename"]:
            print(f"Original filename: {identity['filename']}")
    records = decrypt_log(args.log_path, args.private_key)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} turns to {args.output}")
    else:
        for r in records:
            print(json.dumps(r, ensure_ascii=False))


def _safe_stem(name: str) -> str:
    stem = "".join(c if c.isalnum() else "_" for c in name).strip("_")
    return stem or "unknown"


def _cmd_decrypt_batch(args):
    """Decrypts every *.jsonl.enc file in a folder, one per student
    submission. A bad file (unreadable header, wrong private key,
    corrupted upload) is reported and skipped rather than stopping the
    whole batch -- one broken submission shouldn't block grading the rest
    of the class."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enc_files = sorted(input_dir.glob("*.jsonl.enc"))
    if not enc_files:
        print(f"No .jsonl.enc files found in {input_dir}")
        return

    used_stems = {}
    manifest = []
    for enc_path in enc_files:
        entry = {"source_file": str(enc_path)}
        try:
            identity = read_student_identity(enc_path, args.private_key)
        except Exception as exc:
            entry["status"] = f"error reading header: {exc}"
            manifest.append(entry)
            print(f"[error] {enc_path.name}: {exc}")
            continue
        student_name, student_id, track = identity["student_name"], identity["student_id"], identity["track"]
        entry["student_name"] = student_name
        entry["student_id"] = student_id
        entry["track"] = track
        entry["filename"] = identity["filename"]

        # Every student now uploads three logs, one per track, sharing
        # session_logs/ -- keying the output name on student name alone
        # would collide all three into the same file. track is folded in
        # here for exactly that reason; the numeric suffix stays as a
        # fallback for the genuinely unexpected case (two different
        # students who typed the same name for the same track).
        base = _safe_stem(student_name) if student_name else _safe_stem(enc_path.stem)
        stem = f"{base}_{track}" if track else base
        seen = used_stems.get(stem, 0)
        used_stems[stem] = seen + 1
        out_name = f"{stem}.jsonl" if seen == 0 else f"{stem}_{seen + 1}.jsonl"
        out_path = output_dir / out_name

        try:
            records = decrypt_log(enc_path, args.private_key)
        except Exception as exc:
            entry["status"] = f"error decrypting: {exc}"
            manifest.append(entry)
            print(f"[error] {enc_path.name}: {exc}")
            continue

        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        entry["status"] = "ok"
        entry["output_file"] = str(out_path)
        entry["turn_count"] = len(records)
        manifest.append(entry)
        print(
            f"[ok] {enc_path.name} -> {out_path.name} ({len(records)} turns, "
            f"student: {student_name}, id: {student_id}, track: {track})"
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    ok_count = sum(1 for e in manifest if e.get("status") == "ok")
    print(f"\n{ok_count}/{len(manifest)} decrypted successfully. Manifest written to {manifest_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-keys", help="Generate a new grading RSA keypair.")
    p_gen.add_argument("--private-key", default="grading_private_key.pem", help="Output path for the private key.")
    p_gen.add_argument("--public-key", default=DEFAULT_PUBLIC_KEY_PATH, help="Output path for the public key.")
    p_gen.set_defaults(func=_cmd_generate_keys)

    p_dec = sub.add_parser("decrypt", help="Decrypt a student's .jsonl.enc log.")
    p_dec.add_argument("log_path", help="Path to the student's uploaded .jsonl.enc file.")
    p_dec.add_argument("--private-key", required=True, help="Path to the grading private key.")
    p_dec.add_argument("--output", help="Write decrypted turns as plain JSONL here (default: print to stdout).")
    p_dec.set_defaults(func=_cmd_decrypt)

    p_batch = sub.add_parser("decrypt-batch", help="Decrypt a folder of students' .jsonl.enc logs at once.")
    p_batch.add_argument("input_dir", help="Directory containing student .jsonl.enc submissions.")
    p_batch.add_argument("--private-key", required=True, help="Path to the grading private key.")
    p_batch.add_argument(
        "--output-dir", required=True, help="Directory to write one decrypted .jsonl file per student, plus a manifest.json."
    )
    p_batch.set_defaults(func=_cmd_decrypt_batch)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
