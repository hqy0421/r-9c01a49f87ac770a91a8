"""Resumable, idempotent data collection from a sealed snapshot API.

The collector pages through a *sealed* (immutable) snapshot identified by an
opaque ``snapshot_id`` together with opaque pagination cursors.  Every fetched
page is staged, together with the cursor for the next page, in a single SQLite
transaction so that a crash or a restart simply continues from the last
durable cursor.  Staged data is never exposed as the current panel: a panel
only becomes current through an explicit :meth:`SnapshotCollector.finalize_run`
call, after the whole table has passed the native format checks and the data
contract, and the switch itself is a single atomic transaction.  Any failure
(network, validation, commit) or an expired snapshot therefore leaves the
previous complete version untouched.
"""

__all__ = [
    "SnapshotAPIError",
    "SnapshotExpiredError",
    "MixedSnapshotError",
    "DuplicateCursorError",
    "IdempotencyError",
    "SnapshotValidationError",
    "SnapshotPage",
    "VersionedPanel",
    "RunStatus",
    "DataContract",
    "FormatContract",
    "SnapshotCollector",
    "http_json_fetcher",
    "collect_snapshot",
]

import json
import pickle
import sqlite3
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np
import pandas as pd

from .compat import POLARS_INSTALLED, pl
from .validation import validate_format

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SnapshotAPIError(Exception):
    """Base class for every snapshot collection failure."""


class SnapshotExpiredError(SnapshotAPIError):
    """Raised when the sealed snapshot is no longer available server-side."""


class MixedSnapshotError(SnapshotAPIError):
    """Raised when a page carries a different ``snapshot_id`` than the run."""


class DuplicateCursorError(SnapshotAPIError):
    """Raised when a cursor that was already consumed is requested again."""


class IdempotencyError(SnapshotAPIError):
    """Raised when the same id/time pair is seen with conflicting values."""


class SnapshotValidationError(SnapshotAPIError):
    """Raised when a page or the final table violates the data contract."""


# ---------------------------------------------------------------------------
# Value encoding
#
# Every scalar is reduced to one of the SQLite storage classes (NULL, INTEGER,
# REAL, TEXT) together with a ``kind`` tag and, in the schema table, enough
# metadata to rebuild the exact original pandas dtype.  Booleans, datetimes,
# dates and categoricals need explicit handling because they would otherwise
# collapse onto ints/strings and lose their type on the way back.
# ---------------------------------------------------------------------------

_KIND_INT = "int"
_KIND_FLOAT = "float"
_KIND_BOOL = "bool"
_KIND_DATETIME = "datetime"
_KIND_DATE = "date"
_KIND_STR = "str"
_KIND_CAT = "cat"
_KIND_NULL = "null"


def _series_kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return _KIND_DATETIME
    if isinstance(s.dtype, pd.CategoricalDtype):
        return _KIND_CAT
    if pd.api.types.is_bool_dtype(s.dtype):
        return _KIND_BOOL
    if pd.api.types.is_integer_dtype(s.dtype):
        return _KIND_INT
    if pd.api.types.is_float_dtype(s.dtype):
        return _KIND_FLOAT
    # pandas represents a column of datetime.date values as object dtype
    non_null = s.dropna()
    if s.dtype == object and len(non_null) > 0 and all(
        isinstance(v, date) and not isinstance(v, datetime) for v in non_null
    ):
        return _KIND_DATE
    if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
        return _KIND_STR
    raise SnapshotValidationError(
        f"Column '{s.name}' has unsupported dtype '{s.dtype}'. Supported dtypes "
        "are integer, float, boolean, string/object, datetime, date and categorical."
    )


def _encode_value(value: Any, kind: str) -> Any:
    """Canonical scalar used both for storage and for conflict comparison."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if kind == _KIND_DATETIME:
        if not isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
            raise SnapshotValidationError(
                f"Expected timestamp value, got {type(value).__name__}."
            )
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return str(pd.Timestamp(value).value)
    if kind == _KIND_DATE:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, pd.Timestamp) and value.normalize() == value:
            return value.date().isoformat()
        raise SnapshotValidationError(f"Expected date value, got {value!r}.")
    if kind == _KIND_BOOL:
        return int(bool(value))
    if kind == _KIND_CAT:
        # categoricals are encoded as integer codes in _stage_page, never
        # passed through this generic path
        raise AssertionError("categorical values must be encoded via codes")
    if kind == _KIND_INT:
        return int(value)
    if kind == _KIND_FLOAT:
        f = float(value)
        if np.isnan(f):
            return None
        return f
    return str(value)


def _decode_value(stored: Any, kind: str, meta: Optional[Dict[str, Any]]) -> Any:
    if stored is None:
        return None
    if kind == _KIND_INT:
        return int(stored)
    if kind == _KIND_FLOAT:
        return float(stored)
    if kind == _KIND_BOOL:
        return bool(int(stored))
    if kind == _KIND_DATETIME:
        ts = pd.Timestamp(np.int64(int(stored)))
        tz = (meta or {}).get("tz")
        if tz is not None:
            ts = ts.tz_localize("UTC").tz_convert(tz)
        return ts
    if kind == _KIND_DATE:
        return date.fromisoformat(str(stored))
    if kind == _KIND_CAT:
        return (meta or {}).get("categories", [])[int(stored)]
    return str(stored)

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotPage:
    """One page returned by the snapshot API.

    Attributes:
        snapshot_id: Identifier of the sealed snapshot this page belongs to.
        rows: Panel rows in long format (pandas or polars DataFrame).
        next_cursor: Opaque cursor to request the following page, or ``None``
            when the snapshot has been fully delivered.
        cursor: Cursor that was used to obtain this page (echoed by the
            server; the collector falls back to the requested cursor).
    """

    snapshot_id: str
    rows: Any
    next_cursor: Optional[str]
    cursor: Optional[str] = None


@dataclass(frozen=True)
class VersionedPanel:
    """A readable panel together with its exact provenance.

    Attributes:
        snapshot_id: Sealed snapshot backing this version.
        version: Monotonically increasing, immutable version number.
        df: Panel in the requested native engine (pandas/polars).  Every call
            to :meth:`SnapshotCollector.read` returns a fresh object, so a
            DataFrame handed out earlier is never mutated by a later swap.
    """

    snapshot_id: str
    version: int
    df: Any


class RunStatus:
    STAGING = "staging"
    COMPLETED = "completed"
    FINALIZED = "finalized"
    FAILED = "failed"
    DISCARDED = "discarded"


class DataContract(Protocol):
    """A user supplied validation applied to the complete table."""

    def __call__(self, df: pd.DataFrame) -> None:
        """Raise if ``df`` violates the contract."""


@dataclass
class FormatContract:
    """Native long-format contract checked against the complete table.

    This mirrors :func:`utilsforecast.validation.validate_format`: required
    id/time columns, no nulls in them, integer/timestamp time column and,
    optionally, a numeric target column.
    """

    id_col: str = "unique_id"
    time_col: str = "ds"
    target_col: Optional[str] = "y"

    def __call__(self, df: pd.DataFrame) -> None:
        validate_format(
            df,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class SnapshotCollector:
    """Resumable collector for a sealed snapshot API, backed by SQLite.

    Args:
        db_path: Path of the SQLite staging database.  A single file may hold
            many runs and versions; each collector instance opens its own
            connection.  ``":memory:"`` databases are supported but cannot be
            resumed from another process.
        fetch_page: Callable taking ``(snapshot_id, cursor)`` and returning a
            :class:`SnapshotPage` or a raw JSON-like dict.  Defaults to an
            HTTP/JSON fetcher against ``endpoint``.
        endpoint: HTTP endpoint used by the default fetcher.
        headers: Extra HTTP headers for the default fetcher.
        timeout: Request timeout in seconds for the default fetcher.
        id_col: Name of the series identifier column.
        time_col: Name of the timestamp column.
        contract: Optional callable validated against the *complete* table
            before it is promoted.  Defaults to :class:`FormatContract`.
    """

    def __init__(
        self,
        db_path: str,
        fetch_page: Optional[Callable[[str, Optional[str]], Any]] = None,
        *,
        endpoint: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        id_col: str = "unique_id",
        time_col: str = "ds",
        contract: Optional[DataContract] = None,
    ) -> None:
        self.db_path = db_path
        self.id_col = id_col
        self.time_col = time_col
        self.contract: Optional[DataContract] = (
            contract if contract is not None else FormatContract(id_col, time_col)
        )
        if fetch_page is None:
            if endpoint is None:
                raise ValueError(
                    "Either `fetch_page` or `endpoint` must be provided."
                )
            fetch_page = http_json_fetcher(endpoint, headers=headers, timeout=timeout)
        self._fetch_page = fetch_page
        # check_same_thread=False: the object itself is not thread-safe, but
        # users may hand it across threads (e.g. inside a worker).
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "SnapshotCollector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # -- schema ------------------------------------------------------------

    def _init_schema(self) -> None:
        con = self.con
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs(
                run_id      TEXT PRIMARY KEY,
                snapshot_id TEXT NOT NULL,
                cursor      TEXT,
                status      TEXT NOT NULL,
                n_rows      INTEGER NOT NULL DEFAULT 0,
                error       TEXT
            );
            CREATE TABLE IF NOT EXISTS cursors(
                run_id      TEXT NOT NULL REFERENCES runs(run_id),
                cursor      TEXT NOT NULL,
                seq         INTEGER NOT NULL,
                PRIMARY KEY (run_id, cursor)
            );
            CREATE TABLE IF NOT EXISTS schema_meta(
                run_id   TEXT NOT NULL REFERENCES runs(run_id),
                col      TEXT NOT NULL,
                kind     TEXT NOT NULL,
                dtype    TEXT NOT NULL,
                meta     TEXT NOT NULL,
                PRIMARY KEY (run_id, col)
            );
            CREATE TABLE IF NOT EXISTS staged_keys(
                run_id   TEXT NOT NULL REFERENCES runs(run_id),
                id_val   TEXT NOT NULL,
                time_val TEXT NOT NULL,
                seq      INTEGER NOT NULL,
                PRIMARY KEY (run_id, id_val, time_val)
            );
            CREATE TABLE IF NOT EXISTS staged_values(
                run_id   TEXT NOT NULL REFERENCES runs(run_id),
                id_val   TEXT NOT NULL,
                time_val TEXT NOT NULL,
                col      TEXT NOT NULL,
                kind     TEXT NOT NULL,
                val_int  INTEGER,
                val_real REAL,
                val_text TEXT,
                PRIMARY KEY (run_id, id_val, time_val, col),
                FOREIGN KEY (run_id, id_val, time_val)
                    REFERENCES staged_keys(run_id, id_val, time_val)
            );
            CREATE TABLE IF NOT EXISTS versions(
                version     INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT NOT NULL,
                run_id      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                blob        BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS current(
                id      INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL
            );
            """
        )
        con.commit()

    # -- run lifecycle -----------------------------------------------------

    def start_run(
        self,
        snapshot_id: str,
        run_id: Optional[str] = None,
        *,
        initial_cursor: Optional[str] = None,
    ) -> str:
        """Start (or re-open) a collection run against ``snapshot_id``.

        Re-opening an existing ``run_id`` resumes from the last durable
        cursor.  A run that already failed or was finalized cannot be
        restarted; discard it and start a new run (which necessarily binds a
        fresh snapshot id).
        """
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("`snapshot_id` must be a non-empty string.")
        if run_id is None:
            run_id = f"run-{snapshot_id}"
        row = self.con.execute(
            "SELECT snapshot_id, status, cursor FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            with self.con:
                self.con.execute(
                    "INSERT INTO runs(run_id, snapshot_id, cursor, status) "
                    "VALUES (?, ?, ?, ?)",
                    (run_id, snapshot_id, initial_cursor, RunStatus.STAGING),
                )
        else:
            stored_snapshot, status, _cursor = row
            if stored_snapshot != snapshot_id:
                raise MixedSnapshotError(
                    f"Run '{run_id}' is bound to snapshot '{stored_snapshot}', "
                    f"cannot reuse it with snapshot '{snapshot_id}'."
                )
            if status == RunStatus.FAILED:
                raise SnapshotAPIError(
                    f"Run '{run_id}' previously failed; discard it and start a "
                    "new run instead of silently mixing snapshots."
                )
            if status == RunStatus.FINALIZED:
                raise SnapshotAPIError(
                    f"Run '{run_id}' was already finalized; start a new run for "
                    "a new snapshot."
                )
            if status == RunStatus.DISCARDED:
                raise SnapshotAPIError(f"Run '{run_id}' was discarded.")
        return run_id

    def get_run_cursor(self, run_id: str) -> Optional[str]:
        """Return the durable cursor the next request must use."""
        return self._require_run(run_id)[2]

    def get_run_status(self, run_id: str) -> str:
        return self._require_run(run_id)[1]

    def _require_run(self, run_id: str) -> tuple:
        row = self.con.execute(
            "SELECT snapshot_id, status, cursor FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown run_id '{run_id}'.")
        return row

    def discard_run(self, run_id: str) -> None:
        """Delete all staged state of a run (never touches current version)."""
        with self.con:
            for table in ("staged_values", "staged_keys", "cursors", "schema_meta"):
                self.con.execute(
                    f"DELETE FROM {table} WHERE run_id = ?", (run_id,)
                )
            self.con.execute(
                "UPDATE runs SET status = ?, error = NULL WHERE run_id = ?",
                (RunStatus.DISCARDED, run_id),
            )

    # -- fetching & staging ------------------------------------------------

    @staticmethod
    def _cursor_key(cursor: Optional[str]) -> str:
        return "" if cursor is None else cursor

    def fetch_page(
        self,
        run_id: str,
        cursor: Optional[str] = None,
    ) -> SnapshotPage:
        """Fetch one page and durably stage it with the next cursor.

        The request and the staging share one transaction: either both the
        page rows and the next cursor become durable together, or nothing
        does and the run resumes from the previous cursor on retry.
        """
        snapshot_id, status, current_cursor = self._require_run(run_id)
        if status != RunStatus.STAGING:
            raise SnapshotAPIError(
                f"Run '{run_id}' is in status '{status}', cannot fetch pages."
            )
        if cursor is None:
            cursor = current_cursor
        elif cursor != current_cursor:
            raise DuplicateCursorError(
                f"Run '{run_id}' expects cursor {current_cursor!r}, got {cursor!r}. "
                "Re-requesting a consumed cursor is not allowed."
            )

        if cursor is not None and not isinstance(cursor, str):
            raise DuplicateCursorError(
                f"Cursors must be opaque strings or None, got {type(cursor).__name__}."
            )
        # durable ledger catches duplicate cursors, including after a restart;
        # the current cursor is exempt because its page has not been staged yet
        key = self._cursor_key(cursor)
        if key != self._cursor_key(current_cursor) and self.con.execute(
            "SELECT 1 FROM cursors WHERE run_id = ? AND cursor = ?",
            (run_id, key),
        ).fetchone():
            raise DuplicateCursorError(
                f"Cursor {cursor!r} was already consumed by run '{run_id}'."
            )

        raw = self._fetch_page(snapshot_id, cursor)
        page = self._coerce_page(raw, expected_cursor=cursor)
        if page.snapshot_id != snapshot_id:
            err = MixedSnapshotError(
                f"Page belongs to snapshot '{page.snapshot_id}' but run "
                f"'{run_id}' is bound to '{snapshot_id}'."
            )
            self._mark_failed(run_id, err)
            raise err
        if page.next_cursor is not None and not isinstance(page.next_cursor, str):
            raise SnapshotAPIError(
                "next_cursor must be an opaque string or None, got "
                f"{type(page.next_cursor).__name__}."
            )

        try:
            pdf = self._to_pandas(page.rows)
            self._stage_page(run_id, pdf, page.next_cursor, page.cursor or cursor)
        except SnapshotAPIError as exc:
            # contract/idempotency/cursor failures invalidate the run rather
            # than leaving it silently half-staged under the old snapshot
            self._mark_failed(run_id, exc)
            raise

        if page.next_cursor is None:
            self.con.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?",
                (RunStatus.COMPLETED, run_id),
            )
            self.con.commit()
        return SnapshotPage(
            snapshot_id=snapshot_id,
            rows=pdf,
            next_cursor=page.next_cursor,
            cursor=page.cursor or cursor,
        )

    def _coerce_page(
        self, raw: Any, *, expected_cursor: Optional[str]
    ) -> SnapshotPage:
        if isinstance(raw, SnapshotPage):
            return raw
        if isinstance(raw, dict):
            snap = (
                raw.get("snapshot_id")
                or raw.get("snapshotID")
                or raw.get("snapshot")
            )
            rows = (
                raw.get("rows")
                if "rows" in raw
                else raw.get("data", raw.get("records"))
            )
            nxt = (
                raw.get("next_cursor")
                if "next_cursor" in raw
                else raw.get("nextCursor", raw.get("next"))
            )
            cur = raw.get("cursor", expected_cursor)
            if snap is None or rows is None:
                raise SnapshotAPIError(
                    "Fetcher returned a dict without 'snapshot_id'/'rows' fields."
                )
            return SnapshotPage(
                snapshot_id=str(snap), rows=rows, next_cursor=nxt, cursor=cur
            )
        raise SnapshotAPIError(
            f"Fetcher must return a SnapshotPage or dict, got {type(raw).__name__}."
        )

    def _to_pandas(self, rows: Any) -> pd.DataFrame:
        if isinstance(rows, pd.DataFrame):
            pdf = rows
        elif POLARS_INSTALLED and isinstance(rows, pl.DataFrame):
            pdf = rows.to_pandas()
        elif isinstance(rows, list):
            pdf = pd.DataFrame(rows)
        elif isinstance(rows, dict):
            pdf = pd.DataFrame(rows)
        else:
            raise SnapshotValidationError(
                f"Unsupported rows payload type {type(rows).__name__}."
            )
        if self.id_col not in pdf.columns or self.time_col not in pdf.columns:
            raise SnapshotValidationError(
                f"Page is missing required columns "
                f"'{self.id_col}' and/or '{self.time_col}'; got {list(pdf.columns)}."
            )
        if pdf[self.id_col].isna().any():
            raise SnapshotValidationError(
                f"Id column '{self.id_col}' contains null values."
            )
        if pdf[self.time_col].isna().any():
            raise SnapshotValidationError(
                f"Time column '{self.time_col}' contains null values."
            )
        time_s = pdf[self.time_col]
        if pd.api.types.is_string_dtype(time_s.dtype):
            # JSON payloads carry timestamps as strings; parse them the same
            # way validation.ensure_time_dtype does
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(time_s, errors="raise")
            except (ValueError, TypeError) as exc:
                raise SnapshotValidationError(
                    f"Failed to parse time column '{self.time_col}' as timestamps."
                ) from exc
            pdf[self.time_col] = parsed
            time_s = pdf[self.time_col]
        if not (
            pd.api.types.is_datetime64_any_dtype(time_s.dtype)
            or pd.api.types.is_integer_dtype(time_s.dtype)
        ):
            raise SnapshotValidationError(
                f"Time column '{self.time_col}' must contain timestamps or "
                f"integers, got '{time_s.dtype}'."
            )
        return pdf

    def _stage_page(
        self,
        run_id: str,
        pdf: pd.DataFrame,
        next_cursor: Optional[str],
        used_cursor: Optional[str],
    ) -> None:
        id_col, time_col = self.id_col, self.time_col
        cols = list(pdf.columns)
        kinds: Dict[str, str] = {c: _series_kind(pdf[c]) for c in cols}
        if id_col not in kinds or time_col not in kinds:
            raise SnapshotValidationError(
                f"Required columns '{id_col}'/'{time_col}' missing from page."
            )
        if kinds[id_col] not in (_KIND_STR, _KIND_INT, _KIND_CAT):
            raise SnapshotValidationError(
                f"Id column '{id_col}' must be string, integer or categorical, "
                f"got {kinds[id_col]}."
            )

        con = self.con
        # categorical rows yield the category value under iterrows(); map
        # those values back to their integer codes once per column
        cat_codes = {
            c: {v: i for i, v in enumerate(pdf[c].dtype.categories)}
            for c, k in kinds.items()
            if k == _KIND_CAT
        }
        try:
            # page rows + next cursor commit together, or not at all
            con.execute("BEGIN")
            self._upsert_schema(run_id, pdf, kinds)
            start_seq = con.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM staged_keys WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            for i, (_, row) in enumerate(pdf.iterrows()):
                raw_id = row[id_col]
                if kinds[id_col] == _KIND_CAT and not pd.isna(raw_id):
                    id_key = str(raw_id)
                else:
                    id_key = str(_encode_value(raw_id, kinds[id_col]))
                time_key = str(_encode_value(row[time_col], kinds[time_col]))
                existing = con.execute(
                    "SELECT 1 FROM staged_keys WHERE run_id = ? AND id_val = ? "
                    "AND time_val = ?",
                    (run_id, id_key, time_key),
                ).fetchone()
                if existing is None:
                    con.execute(
                        "INSERT INTO staged_keys(run_id, id_val, time_val, seq) "
                        "VALUES (?, ?, ?, ?)",
                        (run_id, id_key, time_key, start_seq + 1 + i),
                    )
                for col in cols:
                    kind = kinds[col]
                    value = row[col]
                    if kind == _KIND_CAT and not pd.isna(value):
                        value = cat_codes[col][value]
                    self._stage_value(
                        con, run_id, id_key, time_key, col, kind, value
                    )
            # cursor progression: a next cursor that was already consumed
            # means the server is not advancing -> fail loudly instead of
            # looping forever.  The empty key (terminal None sentinel) is
            # exempt because None legitimately marks the end of the stream.
            nxt_key = self._cursor_key(next_cursor)
            if nxt_key and con.execute(
                "SELECT 1 FROM cursors WHERE run_id = ? AND cursor = ?",
                (run_id, nxt_key),
            ).fetchone():
                raise DuplicateCursorError(
                    f"Server returned an already consumed cursor {next_cursor!r}."
                )
            con.execute(
                "UPDATE runs SET cursor = ?, n_rows = ( "
                "SELECT COUNT(*) FROM staged_keys WHERE run_id = ?) "
                "WHERE run_id = ?",
                (next_cursor, run_id, run_id),
            )
            # mark the cursor that produced this page as consumed; the
            # terminal None cursor is recorded too so a resume after the
            # final page cannot re-request it.  IGNORE keeps staging
            # idempotent when the cursor was already durably recorded
            con.execute(
                "INSERT OR IGNORE INTO cursors(run_id, cursor, seq) VALUES (?, ?, "
                "(SELECT COALESCE(MAX(seq), -1) + 1 FROM cursors WHERE run_id = ?))",
                (run_id, self._cursor_key(used_cursor), run_id),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise

    def _upsert_schema(
        self, run_id: str, pdf: pd.DataFrame, kinds: Dict[str, str]
    ) -> None:
        con = self.con
        for col in pdf.columns:
            s = pdf[col]
            kind = kinds[col]
            meta: Dict[str, Any] = {}
            if kind == _KIND_DATETIME:
                # DatetimeTZDtype exposes .unit; plain numpy dtypes need
                # np.datetime_data
                unit = getattr(s.dtype, "unit", None)
                if unit is None:
                    unit = np.datetime_data(np.dtype(s.dtype))[0]
                meta["unit"] = unit
                tz = getattr(s.dtype, "tz", None)
                meta["tz"] = str(tz) if tz is not None else None
            if kind == _KIND_CAT:
                meta["categories"] = list(s.dtype.categories)
            dtype_str = str(s.dtype)
            row = con.execute(
                "SELECT kind, dtype, meta FROM schema_meta "
                "WHERE run_id = ? AND col = ?",
                (run_id, col),
            ).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO schema_meta(run_id, col, kind, dtype, meta) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, col, kind, dtype_str, json.dumps(meta)),
                )
            elif row[0] != kind or row[1] != dtype_str:
                raise SnapshotValidationError(
                    f"Column '{col}' changed dtype across pages: "
                    f"'{row[1]}' -> '{dtype_str}'."
                )
            elif kind == _KIND_CAT:
                old_cats = json.loads(row[2]).get("categories", [])
                if old_cats != meta["categories"]:
                    raise SnapshotValidationError(
                        f"Categorical column '{col}' changed categories across "
                        f"pages: {old_cats!r} -> {meta['categories']!r}."
                    )

    def _stage_value(
        self,
        con: sqlite3.Connection,
        run_id: str,
        id_val: str,
        time_val: str,
        col: str,
        kind: str,
        value: Any,
    ) -> None:
        if kind == _KIND_CAT:
            if pd.isna(value):
                stored, out_kind = None, _KIND_NULL
                val_int = val_real = val_text = None
            else:
                # categorical values arrive as their integer code (see the
                # cat_codes mapping in _stage_page)
                code = int(value)
                stored, out_kind = code, _KIND_CAT
                val_int, val_real, val_text = code, None, None
        else:
            stored = _encode_value(value, kind)
            out_kind = _KIND_NULL if stored is None else kind
            val_int = stored if out_kind in (_KIND_INT, _KIND_BOOL) else None
            val_real = stored if out_kind == _KIND_FLOAT else None
            val_text = (
                stored
                if out_kind in (_KIND_STR, _KIND_DATETIME, _KIND_DATE)
                else None
            )
        existing = con.execute(
            "SELECT kind, val_int, val_real, val_text FROM staged_values "
            "WHERE run_id = ? AND id_val = ? AND time_val = ? AND col = ?",
            (run_id, id_val, time_val, col),
        ).fetchone()
        if existing is not None:
            # idempotent re-delivery: identical values allowed, any mismatch
            # means the sealed snapshot itself is inconsistent
            if tuple(existing) != (out_kind, val_int, val_real, val_text):
                raise IdempotencyError(
                    f"Conflicting values for ({self.id_col}={id_val}, "
                    f"{self.time_col}={time_val}, column='{col}'): "
                    f"stored={existing[1:]!r}, new={(val_int, val_real, val_text)!r}."
                )
            return
        con.execute(
            "INSERT INTO staged_values(run_id, id_val, time_val, col, kind, "
            "val_int, val_real, val_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, id_val, time_val, col, out_kind, val_int, val_real, val_text),
        )

    def _mark_failed(self, run_id: str, exc: Exception) -> None:
        with self.con:
            self.con.execute(
                "UPDATE runs SET status = ?, error = ? WHERE run_id = ?",
                (RunStatus.FAILED, f"{type(exc).__name__}: {exc}", run_id),
            )

    # -- finalization & reads ---------------------------------------------

    def finalize_run(self, run_id: str) -> VersionedPanel:
        """Validate the complete table and atomically promote it.

        The previous current version remains readable throughout: the new
        panel is fully built, validated and serialized *before* the current
        pointer is swapped, and the swap is one atomic statement.
        """
        snapshot_id, status, _ = self._require_run(run_id)
        if status == RunStatus.FINALIZED:
            return self.read()
        if status != RunStatus.COMPLETED:
            raise SnapshotAPIError(
                f"Run '{run_id}' is '{status}', not all pages were collected."
            )
        df = self._reconstruct(run_id)
        if self.contract is not None:
            try:
                self.contract(df)
            except Exception as exc:
                self._mark_failed(run_id, exc)
                raise SnapshotValidationError(
                    f"Data contract validation failed: {exc}"
                ) from exc

        blob = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
        con = self.con
        try:
            con.execute("BEGIN")
            cur = con.execute(
                "INSERT INTO versions(snapshot_id, run_id, created_at, blob) "
                "VALUES (?, ?, ?, ?)",
                (snapshot_id, run_id, datetime.now(timezone.utc).isoformat(), blob),
            )
            new_version = cur.lastrowid
            con.execute(
                "INSERT INTO current(id, version) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET version = excluded.version",
                (new_version,),
            )
            con.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?",
                (RunStatus.FINALIZED, run_id),
            )
            con.commit()
        except Exception:
            con.rollback()
            raise
        return VersionedPanel(
            snapshot_id=snapshot_id, version=new_version, df=df.copy()
        )

    def _reconstruct(self, run_id: str) -> pd.DataFrame:
        schema_rows = self.con.execute(
            "SELECT col, kind, meta FROM schema_meta WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
        if not schema_rows:
            raise SnapshotValidationError(
                f"Run '{run_id}' staged no rows; nothing to reconstruct."
            )
        col_order = [r[0] for r in schema_rows]
        kinds = {r[0]: r[1] for r in schema_rows}
        metas = {r[0]: json.loads(r[2]) for r in schema_rows}
        value_rows = self.con.execute(
            "SELECT id_val, time_val, col, kind, val_int, val_real, val_text "
            "FROM staged_values WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        records: Dict[tuple, Dict[str, Any]] = {}
        for id_val, time_val, col, kind, v_int, v_real, v_text, *_ in value_rows:
            key = (id_val, time_val)
            if kind == _KIND_NULL:
                stored = None
            elif kind in (_KIND_INT, _KIND_BOOL, _KIND_CAT):
                stored = v_int
            elif kind == _KIND_FLOAT:
                stored = v_real
            else:  # str / datetime / date
                stored = v_text
            records.setdefault(key, {})[col] = _decode_value(
                stored, kind, metas.get(col)
            )
        ordering = {
            (r[1], r[2]): r[0]
            for r in self.con.execute(
                "SELECT seq, id_val, time_val FROM staged_keys WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        }
        rows = [records[k] for k in sorted(ordering, key=lambda k: ordering[k])]
        df = pd.DataFrame(rows, columns=col_order)
        df = self._restore_dtypes(df, kinds, metas)
        return df

    def _restore_dtypes(
        self,
        df: pd.DataFrame,
        kinds: Dict[str, str],
        metas: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        for col, kind in kinds.items():
            meta = metas.get(col, {})
            if kind == _KIND_DATETIME:
                unit = meta.get("unit") or "ns"
                tz = meta.get("tz")
                # values were stored as nanosecond integers; rebuild at the
                # original unit (tz-aware values are stored in UTC)
                df[col] = pd.to_datetime(df[col], utc=tz is not None)
                if tz is not None:
                    df[col] = df[col].dt.tz_convert(tz)
                if unit != "ns":
                    df[col] = df[col].astype(f"datetime64[{unit}]" if tz is None
                                             else pd.DatetimeTZDtype(unit=unit, tz=tz))
            elif kind == _KIND_DATE:
                # assign as a plain list so pandas keeps datetime.date
                # objects instead of re-inferring a datetime64 dtype
                df[col] = [
                    None if pd.isna(v)
                    else v if isinstance(v, date) and not isinstance(v, datetime)
                    else pd.Timestamp(v).date()
                    for v in df[col]
                ]
            elif kind == _KIND_BOOL:
                df[col] = df[col].astype("boolean" if df[col].isna().any() else "bool")
            elif kind == _KIND_INT:
                df[col] = df[col].astype("Int64" if df[col].isna().any() else "int64")
            elif kind == _KIND_CAT:
                cats = meta.get("categories", [])
                # _decode_value already mapped codes to category values;
                # rebuild the exact CategoricalDtype from those values
                cat_map = {v: i for i, v in enumerate(cats)}
                codes = np.array(
                    [-1 if pd.isna(v) else cat_map[v] for v in df[col]],
                    dtype="int64",
                )
                df[col] = pd.Categorical.from_codes(codes, categories=cats)
            # str/float columns already have the right inferred dtype
        return df

    def get_current_version(self) -> Optional[int]:
        row = self.con.execute("SELECT version FROM current WHERE id = 1").fetchone()
        return None if row is None else row[0]

    def read(self, engine: str = "pandas") -> VersionedPanel:
        """Return the current complete panel with explicit version info.

        Always returns a fresh DataFrame; previously returned objects are
        never mutated when a newer version is promoted.
        """
        row = self.con.execute(
            "SELECT v.version, v.snapshot_id, v.blob FROM current c "
            "JOIN versions v ON v.version = c.version WHERE c.id = 1"
        ).fetchone()
        if row is None:
            raise SnapshotAPIError("No complete snapshot version has been finalized.")
        version, snapshot_id, blob = row
        df = pickle.loads(blob)
        if engine == "pandas":
            out = df.copy()
        elif engine == "polars":
            if not POLARS_INSTALLED:
                raise SnapshotAPIError(
                    "polars is not installed; read with engine='pandas' or "
                    "install the 'polars' extra."
                )
            out = pl.from_pandas(df)
        else:
            raise ValueError(f"engine must be 'pandas' or 'polars', got {engine!r}.")
        return VersionedPanel(snapshot_id=snapshot_id, version=version, df=out)

    def collect_all(
        self,
        snapshot_id: str,
        run_id: Optional[str] = None,
        *,
        initial_cursor: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> VersionedPanel:
        """Drive a run to completion and promote it in one call.

        Safe to call again with the same ``run_id`` after a crash: collection
        resumes from the durable cursor and promotion is idempotent.
        """
        run_id = self.start_run(
            snapshot_id, run_id, initial_cursor=initial_cursor
        )
        status = self.get_run_status(run_id)
        if status == RunStatus.FINALIZED:
            return self.read()
        pages = 0
        while self.get_run_status(run_id) == RunStatus.STAGING:
            self.fetch_page(run_id)
            pages += 1
            if max_pages is not None and pages >= max_pages:
                raise SnapshotAPIError(
                    f"Reached max_pages={max_pages} before the snapshot completed."
                )
        return self.finalize_run(run_id)


# ---------------------------------------------------------------------------
# Default HTTP/JSON fetcher
# ---------------------------------------------------------------------------


def http_json_fetcher(
    endpoint: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Callable[[str, Optional[str]], SnapshotPage]:
    """Build a fetcher that POSTs ``{snapshot_id, cursor}`` and parses JSON."""

    base_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        base_headers.update(headers)

    def _fetch(snapshot_id: str, cursor: Optional[str]) -> SnapshotPage:
        payload = json.dumps(
            {"snapshot_id": snapshot_id, "cursor": cursor}
        ).encode("utf-8")
        req = urllib.request.Request(endpoint, data=payload, headers=base_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 410:
                raise SnapshotExpiredError(
                    f"Snapshot '{snapshot_id}' has expired (HTTP 410)."
                ) from exc
            detail = exc.read().decode("utf-8", errors="replace")
            raise SnapshotAPIError(
                f"Snapshot API request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SnapshotAPIError(f"Snapshot API request failed: {exc}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SnapshotAPIError(
                f"Snapshot API returned invalid JSON: {exc}"
            ) from exc
        return SnapshotPage(
            snapshot_id=str(
                data.get("snapshot_id") or data.get("snapshotID") or snapshot_id
            ),
            rows=data.get("rows", data.get("data", [])),
            next_cursor=data.get("next_cursor", data.get("nextCursor")),
            cursor=data.get("cursor", cursor),
        )

    return _fetch


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def collect_snapshot(
    db_path: str,
    snapshot_id: str,
    fetch_page: Optional[Callable[[str, Optional[str]], Any]] = None,
    *,
    run_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    engine: str = "pandas",
    id_col: str = "unique_id",
    time_col: str = "ds",
    contract: Optional[DataContract] = None,
) -> VersionedPanel:
    """Open a collector, collect ``snapshot_id`` and return the current panel."""
    with SnapshotCollector(
        db_path,
        fetch_page,
        endpoint=endpoint,
        id_col=id_col,
        time_col=time_col,
        contract=contract,
    ) as collector:
        collector.collect_all(snapshot_id, run_id=run_id)
        return collector.read(engine=engine)
