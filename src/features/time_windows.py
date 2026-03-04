import re
from datetime import datetime
from typing import Optional, Tuple

# HDFS log format (typical):
# 081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-... terminating
LOG_RE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>[A-Z]+)\s+(?P<component>[^:]+):\s+(?P<msg>.*)$"
)

BLOCK_RE = re.compile(r"\bblk_-?\d+\b", re.IGNORECASE)


def parse_hdfs_timestamp(date_str: str, time_str: str) -> datetime:
    """
    HDFS dataset uses yymmdd for date (e.g., 081109 -> 2008-11-09) and hhmmss for time.
    """
    return datetime.strptime(date_str + time_str, "%y%m%d%H%M%S")


def extract_block_id(text: str) -> Optional[str]:
    m = BLOCK_RE.search(text)
    return m.group(0) if m else None


def parse_hdfs_line(line: str) -> Optional[Tuple[datetime, str, str, str, str]]:
    """
    Returns (ts, block_id, level, component, msg) or None if not parseable / no block_id.
    """
    m = LOG_RE.match(line.strip())
    if not m:
        return None

    ts = parse_hdfs_timestamp(m.group("date"), m.group("time"))
    level = m.group("level")
    component = m.group("component").strip()
    msg = m.group("msg")

    block_id = extract_block_id(line)
    if not block_id:
        return None

    return ts, block_id, level, component, msg