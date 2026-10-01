"""Run SQL Lab queries in disposable, resource-limited SQLite workers."""

import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

MAX_SQL_BYTES = 64 * 1024
MAX_ROWS = 1_000
MAX_RESULT_BYTES = 1024 * 1024
QUERY_SECONDS = 2
WORKER_SECONDS = 5
WORKER_MEMORY_BYTES = 256 * 1024 * 1024
# Bound aggregate worker consumption across Streamlit sessions; never queue work.
_WORKER_SLOTS = threading.BoundedSemaphore(2)


class SQLLabError(ValueError):
    """A query was invalid or exceeded the SQL Lab's safety limits."""


def execute_query(database: Path, sql: str) -> dict:
    """Return bounded columns/rows; pandas must only see a successful result."""
    if len(sql) > MAX_SQL_BYTES or len(sql.encode("utf-8")) > MAX_SQL_BYTES:
        raise SQLLabError("SQL exceeds the 64 KiB query limit.")
    if not _WORKER_SLOTS.acquire(blocking=False):
        raise SQLLabError("SQL Lab is busy. Try again shortly.")
    try:
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), str(database)],
                input=sql.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=WORKER_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SQLLabError("Query exceeded the execution time limit.") from exc
        if result.returncode != 0:
            raise SQLLabError("Query worker exceeded its resource limits or failed.")
        payload = json.loads(result.stdout, object_hook=_decode_blob)
        if "error" in payload:
            raise SQLLabError(payload["error"])
        return payload
    finally:
        _WORKER_SLOTS.release()


def _encode_blob(value):
    # SQLite's only non-JSON result type is bytes. Preserve it for pandas.
    if isinstance(value, bytes):
        return {"blob": value.hex()}
    raise TypeError("Unsupported SQL result type")


def _decode_blob(value):
    return bytes.fromhex(value["blob"]) if set(value) == {"blob"} else value


def _authorize(action, arg1, arg2, database, trigger):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and arg2 != "load_extension":
        return sqlite3.SQLITE_OK
    # Includes ATTACH, PRAGMA, writes, schema changes and transaction controls.
    return sqlite3.SQLITE_DENY


def _run_worker(database: Path, sql: str) -> str:
    # Import only in the child. If OS limits are unsupported, fail closed.
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_BYTES, WORKER_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (QUERY_SECONDS + 1, QUERY_SECONDS + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    deadline = time.monotonic() + QUERY_SECONDS
    conn = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.set_authorizer(_authorize)
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        cursor = conn.execute(sql)
        if cursor.description is None:
            raise SQLLabError("Only read-only queries are allowed.")
        columns = json.dumps([column[0] for column in cursor.description])
        prefix = '{"columns":' + columns + ',"rows":['
        size = len(prefix.encode("utf-8")) + 2
        if size > MAX_RESULT_BYTES:
            raise SQLLabError("Query exceeds the 1 MiB result limit.")
        rows = []
        # Never materialize the full cursor. Even a single huge value is confined
        # to the worker's OS memory limit before it reaches this byte check.
        for row in cursor:
            if time.monotonic() >= deadline:
                raise SQLLabError("Query exceeded the execution time limit.")
            if len(rows) == MAX_ROWS:
                raise SQLLabError("Query exceeds the 1,000 row limit. Add a LIMIT clause.")
            encoded = json.dumps(row, default=_encode_blob)
            size += len(encoded.encode("utf-8")) + bool(rows)
            if size > MAX_RESULT_BYTES:
                raise SQLLabError("Query exceeds the 1 MiB result limit.")
            rows.append(encoded)
        return prefix + ",".join(rows) + "]}"
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        query = sys.stdin.buffer.read(MAX_SQL_BYTES + 1)
        if len(query) > MAX_SQL_BYTES:
            raise SQLLabError("SQL exceeds the 64 KiB query limit.")
        output = _run_worker(Path(sys.argv[1]), query.decode("utf-8"))
    except MemoryError:
        output = json.dumps({"error": "Query exceeded the worker memory limit."})
    except Exception as exc:
        output = json.dumps({"error": str(exc)[:1000] or "Query worker failed."})
    sys.stdout.write(output)
