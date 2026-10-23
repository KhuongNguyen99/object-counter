"""Read local settings without executing shell syntax or replacing environment values."""

import os
from pathlib import Path
import re


def load_dotenv(path, environ=None):
    destination = os.environ if environ is None else environ
    source = Path(path)
    if not source.is_file():
        return False
    for number, raw in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid setting name on line {number} of {source.name}")
        if value[:1] in {"'", '"'}:
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or value[end + 1:].strip() and not value[end + 1:].lstrip().startswith("#"):
                raise ValueError(f"Invalid quoted setting on line {number} of {source.name}")
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        destination.setdefault(key, value)
    return True
