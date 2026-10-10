"""File and text helpers. sanitize_filename and the JSON parsing are ported from upstream src/utils/file_utils.py."""

import json
import os
from pathlib import Path

import json5


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a file or directory name."""
    invalid_chars = '<>:"/\\|?*'
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    sanitized = ''.join(char if char.isprintable() else '_' for char in sanitized)
    sanitized = sanitized.strip('. ')
    return sanitized[:200]


def _extract_json_block(text: str) -> str | None:
    """Extract the outermost JSON array or object from text with surrounding prose."""
    for open_char, close_char in [('[', ']'), ('{', '}')]:
        start = text.find(open_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        # Unclosed — return from start to end (truncated JSON, handled by caller)
        return text[start:]
    return None


def _try_fix_truncated_json(text: str) -> str | None:
    """Try to fix truncated JSON by closing open brackets/braces in correct nesting order."""
    stack = []
    in_string = False
    escape_next = False
    for c in text:
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c in '[{':
            stack.append(']' if c == '[' else '}')
        elif c in ']}' and stack:
            stack.pop()
    if not stack:
        return None
    fixed = text.rstrip().rstrip(',')
    if in_string:
        fixed += '"'
    fixed += ''.join(reversed(stack))
    return fixed


def parse_json_response(response: str):
    """Parse the JSON in a model reply, tolerating code fences, surrounding prose and truncation.

    Raises ValueError if nothing parses.
    """
    cleaned = response.replace("```json", "").replace("```", "").strip()
    try:
        return json5.loads(cleaned)
    except ValueError:
        pass
    extracted = _extract_json_block(cleaned)
    if extracted:
        try:
            return json5.loads(extracted)
        except ValueError:
            fixed = _try_fix_truncated_json(extracted)
            if fixed:
                try:
                    return json5.loads(fixed)
                except ValueError:
                    pass
    raise ValueError(f"no JSON in reply ({len(response)} chars): {response[:300]!r}")


def extract_content(response: str) -> str | None:
    """Return the document in the reply's last complete <content>...</content> block, or None if there is none.

    Upstream falls back to the whole reply when the tags are missing, which puts the <scratchpad> plan into the
    training text. Taking the last block also skips a scratchpad that mentions the tag.
    """
    end = response.rfind("</content>")
    start = response.rfind("<content>", 0, end) if end != -1 else -1
    if start == -1:
        return None
    return response[start + len("<content>"):end].strip() or None


def write_atomic(path: Path, text: str) -> None:
    """Write through a temporary file, so an interrupted run never leaves a partial file that a rerun would skip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_json(path: Path, data) -> None:
    write_atomic(path, json.dumps(data, indent=2, ensure_ascii=False))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
