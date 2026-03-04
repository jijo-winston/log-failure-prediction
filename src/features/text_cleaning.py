import re

# Common high-cardinality noise patterns in HDFS logs
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
BLOCK_RE = re.compile(r"\bblk_-?\d+\b", re.IGNORECASE)
HEX_RE = re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE)
PATH_RE = re.compile(r"(/[\w\-.]+)+")
DATE6_RE = re.compile(r"\b\d{6}\b")       # e.g., 081111, 091728
PORT5_RE = re.compile(r"\b\d{5}\b")       # e.g., 50010
LONG_NUM_RE = re.compile(r"\b\d{6,}\b")   # large IDs/sizes like 67108864
WS_RE = re.compile(r"\s+")


def normalize_log_text(text: str) -> str:
    """
    Normalize HDFS log text by masking high-cardinality tokens (IPs, paths, ids, timestamps).
    Keeps useful semantic tokens like error keywords and component names.

    Returns lowercase, whitespace-normalized string.
    """
    if not isinstance(text, str):
        return ""

    s = text.lower()

    # mask patterns
    s = IP_RE.sub(" <ip> ", s)
    s = HEX_RE.sub(" <hex> ", s)
    s = PATH_RE.sub(" <path> ", s)

    # dataset-specific noisy tokens
    s = DATE6_RE.sub(" <dt6> ", s)
    s = PORT5_RE.sub(" <port> ", s)
    s = LONG_NUM_RE.sub(" <num> ", s)

    # block ids appear inside messages; we already group by block_id
    s = BLOCK_RE.sub(" <block> ", s)

    # normalize whitespace
    s = WS_RE.sub(" ", s).strip()
    return s