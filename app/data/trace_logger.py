from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import uuid


TRACE_DIR = Path("storage/traces")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def load_trace(trace_id: str) -> dict:
    """Load one saved trace by UUID without allowing path traversal."""
    try:
        safe_trace_id = str(uuid.UUID(str(trace_id)))
    except (TypeError, ValueError) as exc:
        raise ValueError("trace_id khong hop le") from exc

    trace_path = TRACE_DIR / f"{safe_trace_id}.json"
    if not trace_path.exists():
        raise FileNotFoundError(safe_trace_id)

    return json.loads(trace_path.read_text(encoding="utf-8"))


class RagTrace:
    def __init__(self, question: str):
        self.trace_id = str(uuid.uuid4())
        self.payload = {
            "trace_id": self.trace_id,
            "question": question,
            "created_at": _now(),
            "updated_at": _now(),
            "steps": [],
            "response": None,
        }

    def add_step(self, name: str, output: dict | None = None, input_data: dict | None = None, status: str = "success"):
        timestamp = _now()
        self.payload["steps"].append({
            "name": name,
            "status": status,
            "started_at": timestamp,
            "finished_at": timestamp,
            "input": input_data or {},
            "output": output or {},
        })
        self.payload["updated_at"] = timestamp

    def set_response(self, response: dict):
        self.payload["response"] = response
        self.payload["updated_at"] = _now()

    def save(self) -> str:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        trace_path = TRACE_DIR / f"{self.trace_id}.json"
        trace_path.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return self.trace_id
