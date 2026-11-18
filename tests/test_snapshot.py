import json
import sqlite3
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import numpy as np
import pandas as pd
import pytest

from utilsforecast.compat import POLARS_INSTALLED, pl
from utilsforecast.snapshot import (
    DuplicateCursorError,
    FormatContract,
    IdempotencyError,
    MixedSnapshotError,
    RunStatus,
    SnapshotAPIError,
    SnapshotCollector,
    SnapshotExpiredError,
    SnapshotPage,
    SnapshotValidationError,
    VersionedPanel,
    collect_snapshot,
    http_json_fetcher,
)

if POLARS_INSTALLED:
    engines = ["pandas", "polars"]
else:
    engines = ["pandas"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAPI:
    """Scriptable snapshot API keyed by opaque cursors.

    ``pages`` maps each request cursor to a ``(next_cursor, DataFrame)``
    pair; the initial request always uses ``None``.
    """

    def __init__(self, pages, snapshot_id="snap-1"):
        if isinstance(pages, list):
            # accept the ordered shorthand [(next1, df1), (None, df2)] by
            # threading the cursors together starting from None
            ordered = pages
            pages = {}
            cursor = None
            for next_cursor, rows in ordered:
                pages[cursor] = (next_cursor, rows)
                cursor = next_cursor
        self.pages = pages
        self.snapshot_id = snapshot_id
        self.calls = []
        self.fail_once = {}  # cursor -> exception, raised on first use
        self.expired = False

    def __call__(self, snapshot_id, cursor):
        if self.expired:
            raise SnapshotExpiredError("gone")
        if cursor in self.fail_once:
            raise self.fail_once.pop(cursor)
        self.calls.append(cursor)
        next_cursor, rows = self.pages[cursor]
        return SnapshotPage(snapshot_id, rows, next_cursor, cursor=cursor)


def make_panel(ids, times, y, **extra):
    data = {"unique_id": ids, "ds": pd.to_datetime(times), "y": y}
    data.update(extra)
    return pd.DataFrame(data)


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "snap.db")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_collect_all_paginates_with_cursors(tmp_db):
    p1 = make_panel(["a", "a"], ["2024-01-01", "2024-01-02"], [1.0, 2.0])
    p2 = make_panel(["b"], ["2024-01-01"], [3.0])
    api = FakeAPI([("c1", p1), (None, p2)])

    with SnapshotCollector(tmp_db, api) as c:
        vp = c.collect_all("snap-1")
        assert isinstance(vp, VersionedPanel)
        assert vp.version == 1
        assert vp.snapshot_id == "snap-1"
        assert api.calls == [None, "c1"]
        out = vp.df
        assert len(out) == 3
        assert list(out["unique_id"]) == ["a", "a", "b"]
        assert c.get_run_status("run-snap-1") == RunStatus.FINALIZED
        assert c.get_current_version() == 1


def test_read_returns_explicit_version_and_fresh_copy(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    api = FakeAPI([(None, p1)])
    with SnapshotCollector(tmp_db, api) as c:
        c.collect_all("snap-1")
        r1 = c.read()
        r1.df["y"] = 999.0  # mutate the handed-out panel
        r2 = c.read()
        assert r2.version == 1
        assert r2.df["y"].iloc[0] == 1.0
        assert r1.df is not r2.df


@pytest.mark.parametrize("engine", engines)
def test_read_native_engine_and_dtypes(tmp_db, engine):
    df = pd.DataFrame(
        {
            "unique_id": pd.Categorical(["a", "a", "b"]),
            "ds": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01"]
            ).tz_localize("UTC"),
            "y": [1.5, 2.5, 3.5],
            "n": [10, 20, 30],
            "flag": [True, False, True],
            "d": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1)],
        }
    )
    api = FakeAPI([(None, df)])
    with SnapshotCollector(tmp_db, api) as c:
        c.collect_all("snap-1")
        out = c.read(engine=engine).df
    if engine == "polars":
        assert isinstance(out, pl.DataFrame)
        # polars maps python date objects to its native Date dtype
        assert out.schema["d"] == pl.Date
        out = out.to_pandas()
    else:
        assert isinstance(out, pd.DataFrame)
        assert out["d"].iloc[0] == date(2024, 1, 1)
    # id/time types are preserved
    assert isinstance(out["unique_id"].dtype, pd.CategoricalDtype)
    assert str(out["ds"].dtype) == "datetime64[ns, UTC]"
    assert pd.api.types.is_bool_dtype(out["flag"].dtype)
    assert pd.api.types.is_integer_dtype(out["n"].dtype)
    np.testing.assert_allclose(out["y"], [1.5, 2.5, 3.5])


def test_integer_time_column_preserved(tmp_db):
    df = pd.DataFrame({"unique_id": ["a", "a"], "ds": [1, 2], "y": [1.0, 2.0]})
    api = FakeAPI([(None, df)])
    with SnapshotCollector(tmp_db, api) as c:
        c.collect_all("snap-1")
        out = c.read().df
    assert pd.api.types.is_integer_dtype(out["ds"].dtype)


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def test_resume_after_crash_continues_from_durable_cursor(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    p2 = make_panel(["b"], ["2024-01-01"], [2.0])
    api = FakeAPI([("c1", p1), (None, p2)])

    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-1")
        page = c.fetch_page(rid)
        assert page.next_cursor == "c1"
        assert c.get_run_cursor(rid) == "c1"
        assert c.get_run_status(rid) == RunStatus.STAGING

    # reopen: staged rows are not current, cursor is durable
    api2 = FakeAPI({"c1": (None, p2)})
    with SnapshotCollector(tmp_db, api2) as c:
        assert c.get_current_version() is None
        rid = c.start_run("snap-1")
        assert c.get_run_cursor(rid) == "c1"
        vp = c.collect_all("snap-1", rid)
        # page 1 was never re-fetched: first request uses the durable cursor
        assert api2.calls == ["c1"]
        assert len(vp.df) == 2
        assert c.get_current_version() == 1


def test_page_and_next_cursor_commit_together(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    api = FakeAPI([("c1", p1)])
    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-1")
        c.fetch_page(rid)
        con = sqlite3.connect(tmp_db)
        try:
            n_rows = con.execute(
                "SELECT COUNT(*) FROM staged_values WHERE run_id = ?", (rid,)
            ).fetchone()[0]
            cursor = con.execute(
                "SELECT cursor FROM runs WHERE run_id = ?", (rid,)
            ).fetchone()[0]
            consumed = con.execute(
                "SELECT COUNT(*) FROM cursors WHERE run_id = ?", (rid,)
            ).fetchone()[0]
        finally:
            con.close()
        assert n_rows > 0
        assert cursor == "c1"
        assert consumed == 1  # the None entry cursor got consumed atomically


def test_network_failure_does_not_advance_cursor(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    p2 = make_panel(["b"], ["2024-01-01"], [2.0])
    api = FakeAPI([("c1", p1), (None, p2)])
    api.fail_once = {"c1": SnapshotAPIError("connection reset")}
    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-1")
        c.fetch_page(rid)
        with pytest.raises(SnapshotAPIError, match="connection reset"):
            c.fetch_page(rid)
        # still pointing at c1, no partial state
        assert c.get_run_cursor(rid) == "c1"
        assert c.get_run_status(rid) == RunStatus.STAGING
        # retry succeeds and completes
        vp = c.collect_all("snap-1", rid)
        assert len(vp.df) == 2
        assert api.calls == [None, "c1"]  # failed call not recorded


def test_staged_data_is_not_current(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    api = FakeAPI([("c1", p1)])
    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-1")
        c.fetch_page(rid)
        assert c.get_current_version() is None
        with pytest.raises(SnapshotAPIError, match="No complete"):
            c.read()
        with pytest.raises(SnapshotAPIError, match="not all pages"):
            c.finalize_run(rid)


# ---------------------------------------------------------------------------
# Idempotency / snapshot sealing rules
# ---------------------------------------------------------------------------


def test_duplicate_id_time_same_values_are_idempotent(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    # redeliver the exact same row, then a new one
    p2 = pd.concat(
        [
            make_panel(["a"], ["2024-01-01"], [1.0]),
            make_panel(["a"], ["2024-01-02"], [2.0]),
        ],
        ignore_index=True,
    )
    api = FakeAPI([("c1", p1), (None, p2)])
    with SnapshotCollector(tmp_db, api) as c:
        vp = c.collect_all("snap-1")
        assert len(vp.df) == 2  # duplicate collapsed
        assert list(vp.df["y"]) == [1.0, 2.0]


def test_duplicate_id_time_conflicting_values_rejected(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    p2 = make_panel(["a"], ["2024-01-01"], [9.0])
    api = FakeAPI([("c1", p1), (None, p2)])
    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-1")
        c.fetch_page(rid)
        with pytest.raises(IdempotencyError, match="Conflicting values"):
            c.fetch_page(rid)
        # failed page rolled back: still one row, cursor unchanged
        assert c.get_run_cursor(rid) == "c1"
        assert c.get_current_version() is None


def test_mixed_snapshot_id_in_page_rejected(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    p2 = make_panel(["b"], ["2024-01-01"], [2.0])

    class SwitchingAPI(FakeAPI):
        def __call__(self, snapshot_id, cursor):
            page = super().__call__(snapshot_id, cursor)
            if cursor == "c1":
                return SnapshotPage("snap-OTHER", page.rows, page.next_cursor)
            return page

    api = SwitchingAPI({None: ("c1", p1), "c1": (None, p2)})
    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-1")
        c.fetch_page(rid)
        with pytest.raises(MixedSnapshotError):
            c.fetch_page(rid)
        assert c.get_run_status(rid) == RunStatus.FAILED


def test_cannot_reopen_run_under_another_snapshot(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    api = FakeAPI([("c1", p1)])
    with SnapshotCollector(tmp_db, api) as c:
        c.start_run("snap-1", run_id="r")
        c.fetch_page("r")
    with SnapshotCollector(tmp_db, api) as c:
        with pytest.raises(MixedSnapshotError):
            c.start_run("snap-2", run_id="r")


def test_repeated_cursor_request_rejected(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    p2 = make_panel(["b"], ["2024-01-01"], [2.0])
    api = FakeAPI([("c1", p1), ("c2", p2)])
    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-1")
        c.fetch_page(rid)  # None -> c1
        c.fetch_page(rid)  # c1 -> c2
        with pytest.raises(DuplicateCursorError):
            c.fetch_page(rid, cursor="c1")  # c1 was consumed


def test_server_returning_consumed_cursor_rejected(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    p2 = make_panel(["b"], ["2024-01-01"], [2.0])
    p3 = make_panel(["c"], ["2024-01-01"], [3.0])

    class LoopAPI(FakeAPI):
        def __call__(self, snapshot_id, cursor):
            if cursor == "c2":
                # hand back an already consumed cursor instead of advancing
                return SnapshotPage(snapshot_id, p3, "c1")
            return super().__call__(snapshot_id, cursor)

    api = LoopAPI({None: ("c1", p1), "c1": ("c2", p2), "c2": (None, p3)})
    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-1")
        c.fetch_page(rid)  # None -> c1
        c.fetch_page(rid)  # c1 -> c2
        with pytest.raises(DuplicateCursorError, match="already consumed"):
            c.fetch_page(rid)
        assert c.get_run_cursor(rid) == "c2"  # not advanced


# ---------------------------------------------------------------------------
# Atomic version switching
# ---------------------------------------------------------------------------


def test_successful_finalize_switches_version_atomically(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    api = FakeAPI([(None, p1)])
    with SnapshotCollector(tmp_db, api) as c:
        c.collect_all("snap-1", run_id="r1")
        v1 = c.read()
        assert v1.version == 1

    p2 = make_panel(["a", "a"], ["2024-01-01", "2024-01-02"], [5.0, 6.0])
    api2 = FakeAPI([(None, p2)], snapshot_id="snap-2")
    with SnapshotCollector(tmp_db, api2) as c:
        c.collect_all("snap-2", run_id="r2")
        v2 = c.read()
        assert v2.version == 2
        assert v2.snapshot_id == "snap-2"
        assert len(v2.df) == 2
        # old DataFrame object is untouched by the swap
        assert len(v1.df) == 1
        assert v1.df["y"].iloc[0] == 1.0
        assert v1.snapshot_id == "snap-1"


def test_contract_failure_keeps_previous_version(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    FakeAPI([(None, p1)])
    with SnapshotCollector(tmp_db, FakeAPI([(None, p1)])) as c:
        c.collect_all("snap-1", run_id="r1")

    bad = pd.DataFrame(
        {"unique_id": ["a"], "ds": pd.to_datetime(["2024-01-01"]), "y": ["notnum"]}
    )
    with SnapshotCollector(tmp_db, FakeAPI([(None, bad)], "snap-2")) as c:
        with pytest.raises(SnapshotValidationError, match="contract"):
            c.collect_all("snap-2", run_id="r2")
        cur = c.read()
        assert cur.version == 1
        assert cur.snapshot_id == "snap-1"
        assert c.get_run_status("r2") == RunStatus.FAILED


def test_commit_failure_keeps_previous_version(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    with SnapshotCollector(tmp_db, FakeAPI([(None, p1)])) as c:
        c.collect_all("snap-1", run_id="r1")

    p2 = make_panel(["b"], ["2024-01-01"], [2.0])
    with SnapshotCollector(tmp_db, FakeAPI({None: (None, p2)}, "snap-2")) as c:
        c.start_run("snap-2", run_id="r2")
        c.fetch_page("r2")
        # simulate a commit-time storage failure by wrapping the connection:
        # the version insert blows up while everything else stays usable
        real_con = c.con

        class FailingConnection:
            def __getattr__(self, name):
                return getattr(real_con, name)

            def execute(self, *args, **kwargs):
                if args and "INSERT INTO versions" in str(args[0]):
                    raise sqlite3.OperationalError("disk I/O error")
                return real_con.execute(*args, **kwargs)

        c.con = FailingConnection()
        with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
            c.finalize_run("r2")
        c.con = real_con
    # reopen: previous complete version still served, no phantom current row
    with SnapshotCollector(tmp_db, FakeAPI([(None, p2)], "snap-2")) as c:
        cur = c.read()
        assert cur.version == 1
        assert cur.snapshot_id == "snap-1"


def test_expired_snapshot_keeps_previous_version(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    with SnapshotCollector(tmp_db, FakeAPI([(None, p1)])) as c:
        c.collect_all("snap-1", run_id="r1")

    api = FakeAPI([(None, p1)], "snap-2")
    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-2", run_id="r2")
        api.expired = True
        with pytest.raises(SnapshotExpiredError):
            c.fetch_page(rid)
        assert c.read().version == 1


def test_custom_contract_runs_on_full_table(tmp_db):
    df = make_panel(["a", "a"], ["2024-01-01", "2024-01-02"], [1.0, -2.0])

    def nonnegative(pdf):
        if (pdf["y"] < 0).any():
            raise ValueError("targets must be non-negative")

    with SnapshotCollector(tmp_db, FakeAPI([(None, df)]), contract=nonnegative) as c:
        with pytest.raises(SnapshotValidationError, match="non-negative"):
            c.collect_all("snap-1")
        assert c.get_current_version() is None


def test_format_contract_rejects_missing_columns(tmp_db):
    df = pd.DataFrame({"unique_id": ["a"], "ds": pd.to_datetime(["2024-01-01"])})
    with SnapshotCollector(
        tmp_db, FakeAPI([(None, df)]), contract=FormatContract()
    ) as c:
        with pytest.raises(SnapshotValidationError):
            c.collect_all("snap-1")


# ---------------------------------------------------------------------------
# Validation edge cases
# ---------------------------------------------------------------------------


def test_dtype_change_across_pages_rejected(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    p2 = pd.DataFrame(
        {"unique_id": ["b"], "ds": pd.to_datetime(["2024-01-01"]), "y": ["x"]}
    )
    api = FakeAPI([("c1", p1), (None, p2)])
    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-1")
        c.fetch_page(rid)
        with pytest.raises(SnapshotValidationError, match="changed dtype"):
            c.fetch_page(rid)


def test_null_id_or_time_rejected(tmp_db):
    df = pd.DataFrame(
        {"unique_id": [None], "ds": pd.to_datetime(["2024-01-01"]), "y": [1.0]}
    )
    with SnapshotCollector(tmp_db, FakeAPI([(None, df)])) as c:
        with pytest.raises(SnapshotValidationError, match="null values"):
            c.collect_all("snap-1")


def test_unparseable_time_column_rejected(tmp_db):
    df = pd.DataFrame(
        {"unique_id": ["a", "b"], "ds": ["not-a-date", "nope"], "y": [1.0, 2.0]}
    )
    with SnapshotCollector(tmp_db, FakeAPI({None: (None, df)})) as c:
        with pytest.raises(SnapshotValidationError, match="parse time column"):
            c.collect_all("snap-1")


def test_string_time_column_is_parsed(tmp_db):
    df = pd.DataFrame(
        {"unique_id": ["a"], "ds": ["2024-01-01"], "y": [1.0]}
    )
    with SnapshotCollector(tmp_db, FakeAPI({None: (None, df)})) as c:
        vp = c.collect_all("snap-1")
        assert pd.api.types.is_datetime64_any_dtype(vp.df["ds"].dtype)


def test_polars_input_pages_round_trip(tmp_db):
    if not POLARS_INSTALLED:
        pytest.skip("polars not installed")
    pdf = make_panel(["a", "b"], ["2024-01-01", "2024-01-01"], [1.0, 2.0])

    def fetch(sid, cursor):
        return SnapshotPage(sid, pl.from_pandas(pdf), None)

    with SnapshotCollector(tmp_db, fetch) as c:
        vp = c.collect_all("snap-1")
        assert len(vp.df) == 2
        assert c.read(engine="polars").df.shape == (2, 3)


def test_dict_payloads_accepted(tmp_db):
    payload = [
        {"unique_id": "a", "ds": pd.Timestamp("2024-01-01"), "y": 1.0},
        {"unique_id": "a", "ds": pd.Timestamp("2024-01-02"), "y": 2.0},
    ]

    def fetch(sid, cursor):
        return {"snapshot_id": sid, "rows": payload, "next_cursor": None}

    with SnapshotCollector(tmp_db, fetch) as c:
        vp = c.collect_all("snap-1")
        assert len(vp.df) == 2


def test_new_column_in_later_page_is_backfilled(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    p2 = make_panel(["b"], ["2024-01-01"], [2.0], exog=[7.0])
    api = FakeAPI({None: ("c1", p1), "c1": (None, p2)})
    with SnapshotCollector(tmp_db, api) as c:
        vp = c.collect_all("snap-1")
        assert "exog" in vp.df.columns
        # earlier rows have nulls for the late-arriving column
        first = vp.df.sort_values("unique_id").iloc[0]
        assert pd.isna(first["exog"])


def test_categorical_category_drift_rejected(tmp_db):
    p1 = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "g": pd.Categorical(["x"], categories=["x", "y"]),
            "y": [1.0],
        }
    )
    p2 = pd.DataFrame(
        {
            "unique_id": ["b"],
            "ds": pd.to_datetime(["2024-01-01"]),
            "g": pd.Categorical(["z"], categories=["x", "z"]),
            "y": [2.0],
        }
    )
    api = FakeAPI({None: ("c1", p1), "c1": (None, p2)})
    with SnapshotCollector(tmp_db, api) as c:
        rid = c.start_run("snap-1")
        c.fetch_page(rid)
        with pytest.raises(SnapshotValidationError, match="changed categories"):
            c.fetch_page(rid)


def test_row_order_across_pages_is_preserved(tmp_db):
    p1 = make_panel(["a"], ["2024-01-03"], [1.0])
    p2 = make_panel(["b"], ["2024-01-01"], [2.0])
    p3 = make_panel(["c"], ["2024-01-02"], [3.0])
    api = FakeAPI({None: ("c1", p1), "c1": ("c2", p2), "c2": (None, p3)})
    with SnapshotCollector(tmp_db, api) as c:
        vp = c.collect_all("snap-1")
        assert list(vp.df["unique_id"]) == ["a", "b", "c"]


def test_non_nanosecond_tz_aware_time_preserved(tmp_db):
    df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": pd.date_range("2024-01-01", periods=2, freq="h", tz="America/Lima"),
            "y": [1.0, 2.0],
        }
    )
    df["ds"] = df["ds"].astype("datetime64[us, America/Lima]")
    with SnapshotCollector(tmp_db, FakeAPI({None: (None, df)})) as c:
        out = c.collect_all("snap-1").df
    assert str(out["ds"].dtype) == "datetime64[us, America/Lima]"


# ---------------------------------------------------------------------------
# Discard / lifecycle
# ---------------------------------------------------------------------------


def test_discard_run_clears_staging_but_keeps_current(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    with SnapshotCollector(tmp_db, FakeAPI([(None, p1)])) as c:
        c.collect_all("snap-1", run_id="r1")

    with SnapshotCollector(tmp_db, FakeAPI([("c1", p1)])) as c:
        rid = c.start_run("snap-9", run_id="r9")
        c.fetch_page(rid)
        c.discard_run(rid)
        assert c.get_run_status(rid) == RunStatus.DISCARDED
        n = c.con.execute(
            "SELECT COUNT(*) FROM staged_values WHERE run_id = ?", (rid,)
        ).fetchone()[0]
        assert n == 0
        assert c.read().version == 1


def test_finalize_is_idempotent(tmp_db):
    p1 = make_panel(["a"], ["2024-01-01"], [1.0])
    with SnapshotCollector(tmp_db, FakeAPI([(None, p1)])) as c:
        c.collect_all("snap-1", run_id="r1")
        again = c.finalize_run("r1")
        assert again.version == 1


# ---------------------------------------------------------------------------
# HTTP fetcher against a real local server
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    store = {}
    expired_snapshots = set()

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(length))
        sid = req["snapshot_id"]
        if sid in self.expired_snapshots:
            self.send_response(410)
            self.end_headers()
            return
        page = self.store[sid](req["cursor"])
        body = json.dumps(page, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_http_json_fetcher_end_to_end(tmp_db, http_server):
    rows1 = [
        {"unique_id": "a", "ds": "2024-01-01", "y": 1.0},
        {"unique_id": "a", "ds": "2024-01-02", "y": 2.0},
    ]

    def pages(cursor):
        if cursor is None:
            return {"snapshot_id": "snap-http", "rows": rows1, "nextCursor": "c1"}
        return {
            "snapshot_id": "snap-http",
            "rows": [{"unique_id": "b", "ds": "2024-01-01", "y": 3.0}],
            "nextCursor": None,
        }

    _Handler.store["snap-http"] = pages
    fetcher = http_json_fetcher(http_server)
    with SnapshotCollector(tmp_db, fetcher) as c:
        vp = c.collect_all("snap-http")
        assert len(vp.df) == 3


def test_http_fetcher_maps_410_to_expired(tmp_db, http_server):
    _Handler.expired_snapshots.add("snap-gone")
    fetcher = http_json_fetcher(http_server)
    with SnapshotCollector(tmp_db, fetcher) as c:
        with pytest.raises(SnapshotExpiredError):
            fetcher("snap-gone", None)
    _Handler.expired_snapshots.discard("snap-gone")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def test_collect_snapshot_convenience(tmp_path):
    db = str(tmp_path / "x.db")
    df = make_panel(["a"], ["2024-01-01"], [1.0])
    vp = collect_snapshot(db, "snap-1", FakeAPI([(None, df)]))
    assert vp.snapshot_id == "snap-1"
    assert isinstance(vp.df, pd.DataFrame)


def test_collector_requires_fetcher_or_endpoint(tmp_db):
    with pytest.raises(ValueError, match="endpoint"):
        SnapshotCollector(tmp_db)
