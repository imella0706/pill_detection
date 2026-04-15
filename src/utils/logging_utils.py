from __future__ import annotations

import atexit
import datetime as dt
import re
import sys
from pathlib import Path
from typing import TextIO


def _sanitize_name(name: str) -> str:
    """임의의 실행 이름을 파일명으로 안전하게 쓸 수 있는 형태로 정리한다."""
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return clean[:120] if clean else "run"


class TeeStream:
    """하나의 출력 내용을 여러 스트림(콘솔 + 파일)에 동시에 기록한다."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        """설정된 모든 스트림에 동일한 데이터를 기록한다."""
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        """버퍼 손실 방지를 위해 모든 스트림을 flush 한다."""
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        """하위 스트림 중 하나라도 TTY면 TTY 동작을 유지한다."""
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


class LogSession:
    """런타임 로그 세션 생명주기를 관리하고 원래 stdout/stderr를 복구한다."""

    def __init__(
        self,
        log_path: Path,
        file_stream: TextIO,
        orig_stdout: TextIO,
        orig_stderr: TextIO,
    ) -> None:
        self.log_path = log_path
        self.file_stream = file_stream
        self.orig_stdout = orig_stdout
        self.orig_stderr = orig_stderr
        self.stopped = False

    def stop(self) -> None:
        """로그 파일을 닫고 콘솔 출력 스트림을 원래 상태로 복구한다."""
        if self.stopped:
            return
        self.stopped = True
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
        self.file_stream.flush()
        self.file_stream.close()


def start_run_logging(
    project_root: Path,
    category: str,
    run_name: str,
    enabled: bool = True,
) -> LogSession | None:
    """
    단일 실행(run)에 대한 tee 로깅을 시작한다.

    로그 저장 경로:
      logs/<category>/<run_name>_<timestamp>.log
    """
    if not enabled:
        return None

    safe_name = _sanitize_name(run_name)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "logs" / category
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{safe_name}_{timestamp}.log"

    file_stream = log_path.open("w", encoding="utf-8")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = TeeStream(orig_stdout, file_stream)
    sys.stderr = TeeStream(orig_stderr, file_stream)

    session = LogSession(log_path, file_stream, orig_stdout, orig_stderr)
    atexit.register(session.stop)
    print(f"[LOG] saving runtime log to: {log_path}")
    return session
