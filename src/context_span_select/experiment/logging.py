from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class ExperimentSpec:
    """Required metadata for every experiment run."""

    experiment_id: str
    title: str
    description: str
    hypothesis: str
    method: str
    dataset: str
    metrics: list[str]
    budgets: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    author: str = "css"
    notes: str = ""

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("experiment_id", self.experiment_id),
                ("title", self.title),
                ("description", self.description),
                ("hypothesis", self.hypothesis),
                ("method", self.method),
                ("dataset", self.dataset),
            )
            if not str(value).strip()
        ]
        if missing:
            raise ValueError(f"experiment description incomplete: {', '.join(missing)}")
        if len(self.description.strip()) < 80:
            raise ValueError("description must be at least 80 characters")
        if not self.metrics:
            raise ValueError("metrics list must be non-empty")


class ExperimentRun:
    """Strict logging: console + run.log + events.jsonl + metrics.json + summary.json."""

    def __init__(self, root: Path, spec: ExperimentSpec, config: dict[str, Any] | None = None):
        spec.validate()
        self.spec = spec
        self.config = config or {}
        self.started = time.perf_counter()
        self.run_id = f"{spec.experiment_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        self.dir = root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "artifacts").mkdir()
        self.events_path = self.dir / "events.jsonl"
        self.log_path = self.dir / "run.log"
        self._write_json(self.dir / "description.json", asdict(spec))
        self._write_json(self.dir / "config.json", self.config)
        (self.dir / "DESCRIPTION.md").write_text(self._markdown(), encoding="utf-8")

        self.logger = logging.getLogger(f"css.exp.{self.run_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        self.event(
            "run_start",
            {
                "run_id": self.run_id,
                "experiment_id": spec.experiment_id,
                "dir": str(self.dir),
            },
        )
        self.logger.info("RUN %s", self.run_id)
        self.logger.info("TITLE %s", spec.title)
        self.logger.info("DESC %s", spec.description.replace("\n", " "))

    def _markdown(self) -> str:
        s = self.spec
        budgets = ", ".join(map(str, s.budgets)) if s.budgets else "—"
        return (
            f"# {s.title}\n\n"
            f"- **experiment_id:** `{s.experiment_id}`\n"
            f"- **run_id:** `{self.run_id}`\n"
            f"- **method:** {s.method}\n"
            f"- **dataset:** {s.dataset}\n"
            f"- **metrics:** {', '.join(s.metrics)}\n"
            f"- **budgets:** {budgets}\n"
            f"- **tags:** {', '.join(s.tags) if s.tags else '—'}\n\n"
            f"## Hypothesis\n\n{s.hypothesis}\n\n"
            f"## Description\n\n{s.description}\n\n"
            f"## Notes\n\n{s.notes or '—'}\n"
        )

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    def event(self, kind: str, payload: dict[str, Any] | None = None, level: str = "INFO") -> None:
        row = {
            "ts": _utc_now(),
            "elapsed_s": round(time.perf_counter() - self.started, 3),
            "kind": kind,
            "level": level,
            "payload": payload or {},
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        msg = f"{kind} {json.dumps(payload or {}, ensure_ascii=False, default=str)}"
        getattr(self.logger, level.lower(), self.logger.info)(msg)

    def artifact(self, name: str, obj: Any) -> Path:
        path = self.dir / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, (dict, list)):
            self._write_json(path if path.suffix else path.with_suffix(".json"), obj)
            out = path if path.suffix else path.with_suffix(".json")
        else:
            path.write_text(str(obj), encoding="utf-8")
            out = path
        self.event("artifact_written", {"path": str(out.relative_to(self.dir))})
        return out

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        self._write_json(self.dir / "metrics.json", metrics)
        self.event("metrics", metrics)

    def fail(self, exc: BaseException) -> None:
        self.event(
            "run_fail",
            {"error": str(exc), "traceback": traceback.format_exc()},
            level="ERROR",
        )
        self.logger.exception("FAILED: %s", exc)

    def finish(self, summary: dict[str, Any] | None = None) -> Path:
        payload = {
            "run_id": self.run_id,
            "experiment_id": self.spec.experiment_id,
            "elapsed_s": round(time.perf_counter() - self.started, 3),
            "finished_at": _utc_now(),
            "summary": summary or {},
        }
        path = self.dir / "summary.json"
        self._write_json(path, payload)
        self.event("run_finish", payload)
        self.logger.info("DONE elapsed=%.1fs dir=%s", payload["elapsed_s"], self.dir)
        return path
