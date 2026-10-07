"""End-to-end tests for session sealing (HTTP archival)."""
from __future__ import annotations

import json
import multiprocessing
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Operation
from app.sessions import (
    RevisionConflictError,
    SessionSealedError,
    SessionStore,
)


@pytest.fixture()
def store(tmp_path):
    return SessionStore(str(tmp_path / "sessions"))


@pytest.fixture()
def client(store):
    app.state.session_store = store
    return TestClient(app)


def op(actor, seq, version, path, action, value=None, index=None):
    body = {
        "actor": actor,
        "seq": seq,
        "version": version,
        "path": path,
        "action": action,
        "value": value,
    }
    if index is not None:
        body["index"] = index
    return body


def create_session(client, document=None, version=None):
    resp = client.post(
        "/sessions", json={"document": document, "version": version or {}}
    )
    assert resp.status_code == 200
    return resp.json()["session_id"]


def submit(client, session_id, expected_revision, operations):
    return client.post(
        f"/sessions/{session_id}/batches",
        json={"expected_revision": expected_revision, "operations": operations},
    )


def seal(client, session_id, expected_revision):
    return client.post(
        f"/sessions/{session_id}/seal",
        json={"expected_revision": expected_revision},
    )


def replace(client, session_id, expected_revision, actor, path, value):
    return client.post(
        f"/sessions/{session_id}/replace",
        json={
            "expected_revision": expected_revision,
            "actor": actor,
            "path": path,
            "value": value,
        },
    )


# ---------------------------------------------------------------------------
# seal basics
# ---------------------------------------------------------------------------


def test_unsealed_session_reports_sealed_false(client):
    sid = create_session(client, {"a": 1})
    body = client.get(f"/sessions/{sid}").json()
    assert body["sealed"] is False


def test_seal_returns_current_state_with_sealed_true(client):
    sid = create_session(
        client, {"title": "draft", "keep": 1, "gone": 2}
    )
    # alice and bob concurrently set /title (that conflict stays visible
    # in the final document); zoe causally deletes the initially present
    # /gone, which becomes a tombstone
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/title", "set", "a")])
    submit(
        client,
        sid,
        1,
        [op("bob", 1, {"bob": 1}, "/title", "set", "b")],
    )
    submit(
        client,
        sid,
        2,
        [
            op(
                "zoe",
                1,
                {"alice": 1, "bob": 1, "zoe": 1},
                "/gone",
                "delete",
            ),
        ],
    )

    resp = seal(client, sid, 3)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == sid
    assert body["revision"] == 3  # sealing does not bump the revision
    assert body["sealed"] is True
    assert body["document"] == {"title": "b", "keep": 1}
    assert body["version"] == {"alice": 1, "bob": 1, "zoe": 1}
    assert len(body["conflicts"]) == 1
    assert body["conflicts"][0]["path"] == "/title"
    assert body["tombstones"] == ["/gone"]


def test_seal_at_revision_zero(client):
    sid = create_session(client, {"v": 0})
    resp = seal(client, sid, 0)
    assert resp.status_code == 200
    assert resp.json()["revision"] == 0
    assert resp.json()["sealed"] is True
    assert resp.json()["document"] == {"v": 0}


def test_seal_writes_no_revision_log_entry_or_history_boundary(client, store):
    sid = create_session(client, {"n": 0})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", 1)])
    raw_before = json.loads(store._path(sid).read_text(encoding="utf-8"))
    assert raw_before["revision"] == 1
    boundaries_before = [b["revision"] for b in raw_before["history"]]

    resp = seal(client, sid, 1)
    assert resp.status_code == 200

    raw_after = json.loads(store._path(sid).read_text(encoding="utf-8"))
    assert raw_after["revision"] == 1
    assert raw_after["sealed"] is True
    assert raw_after["log"] == raw_before["log"]
    assert [b["revision"] for b in raw_after["history"]] == boundaries_before


def test_get_state_shows_sealed_after_seal(client):
    sid = create_session(client, {})
    seal(client, sid, 0)
    body = client.get(f"/sessions/{sid}").json()
    assert body["sealed"] is True
    assert body["revision"] == 0


def test_seal_unknown_session_returns_404(client):
    missing = "0" * 32
    resp = seal(client, missing, 0)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_seal_missing_expected_revision_returns_422(client):
    sid = create_session(client, {})
    resp = client.post(f"/sessions/{sid}/seal", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# optimistic concurrency on seal
# ---------------------------------------------------------------------------


def test_stale_seal_returns_409_and_leaves_session_unsealed(client):
    sid = create_session(client, {})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    # another commit landed first: sealing against the old revision 0 fails
    resp = seal(client, sid, 0)
    assert resp.status_code == 409
    assert "expected revision 0" in resp.json()["detail"]
    state = client.get(f"/sessions/{sid}").json()
    assert state["sealed"] is False
    assert state["revision"] == 1
    # and the session is still writable at the current revision
    follow = submit(
        client, sid, 1, [op("bob", 1, {"bob": 1}, "/y", "set", 2)]
    )
    assert follow.status_code == 200


def test_future_expected_revision_seal_returns_409(client):
    sid = create_session(client, {})
    resp = seal(client, sid, 9)
    assert resp.status_code == 409
    assert client.get(f"/sessions/{sid}").json()["sealed"] is False


def test_seal_is_idempotent_at_the_same_revision(client, store):
    sid = create_session(client, {"n": 0})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", 1)])
    first = seal(client, sid, 1)
    assert first.status_code == 200
    raw = store._path(sid).read_text(encoding="utf-8")

    second = seal(client, sid, 1)
    assert second.status_code == 200
    assert second.json() == first.json()
    # an idempotent re-seal does not even rewrite the record
    assert store._path(sid).read_text(encoding="utf-8") == raw


def test_reseal_after_someone_elses_commit_is_409(client):
    sid = create_session(client, {})
    assert seal(client, sid, 0).status_code == 200
    # the revision cannot move on a sealed session, but a stale guard
    # still conflicts rather than silently re-confirming
    resp = seal(client, sid, 1)
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# writes after sealing are all 409 and change nothing
# ---------------------------------------------------------------------------


def test_batch_after_seal_returns_409_even_for_retransmission(client, store):
    batch = [op("alice", 1, {"alice": 1}, "/x", "set", 1)]
    sid = create_session(client, {})
    submit(client, sid, 0, batch)
    seal(client, sid, 1)
    raw = store._path(sid).read_text(encoding="utf-8")

    # a genuinely new batch is rejected
    fresh = submit(
        client, sid, 1, [op("bob", 1, {"bob": 1}, "/y", "set", 2)]
    )
    assert fresh.status_code == 409
    assert "sealed" in fresh.json()["detail"]
    # an exact retransmission, normally a 200 no-op, is rejected too
    retry = submit(client, sid, 1, batch)
    assert retry.status_code == 409
    assert "sealed" in retry.json()["detail"]
    # a stale revision is rejected regardless of the revision mismatch
    stale = submit(
        client, sid, 0, [op("carol", 1, {"carol": 1}, "/z", "set", 3)]
    )
    assert stale.status_code == 409

    # nothing on disk changed: document, log, revision, history
    assert store._path(sid).read_text(encoding="utf-8") == raw
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["document"] == {"x": 1}
    assert state["sealed"] is True


def test_replace_after_seal_returns_409_even_for_zero_hit(client, store):
    sid = create_session(client, {"items": [1, 2]})
    submit(
        client,
        sid,
        0,
        [op("alice", 1, {"alice": 1}, "/items/0", "set", 10)],
    )
    seal(client, sid, 1)
    raw = store._path(sid).read_text(encoding="utf-8")

    # a matching replacement would normally commit
    hit = replace(client, sid, 1, "bob", "$.items[*]", 0)
    assert hit.status_code == 409
    assert "sealed" in hit.json()["detail"]
    # a zero-hit query, normally a 200 no-op, is rejected as well
    miss = replace(client, sid, 1, "bob", "$.absent", 0)
    assert miss.status_code == 409
    assert "sealed" in miss.json()["detail"]
    # and a stale guard never bypasses the seal
    stale = replace(client, sid, 0, "carol", "$.items", [])
    assert stale.status_code == 409

    assert store._path(sid).read_text(encoding="utf-8") == raw
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["document"] == {"items": [10, 2]}


def test_session_sealed_error_is_a_revision_conflict():
    assert issubclass(SessionSealedError, RevisionConflictError)


# ---------------------------------------------------------------------------
# reads and incremental change paging keep working after sealing
# ---------------------------------------------------------------------------


def test_changes_paging_still_works_after_seal(client):
    sid = create_session(client, {"n": 0})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", 1)])
    submit(client, sid, 1, [op("alice", 2, {"alice": 2}, "/n", "set", 2)])
    seal(client, sid, 2)

    resp = client.post(
        f"/sessions/{sid}/changes",
        json={
            "after_revision": 0,
            "subscriptions": [{"subscriber_id": "s", "include": ["/**"]}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_revision"] == 2
    assert [c["revision"] for c in body["changes"]] == [1, 2]

    # cursor at the sealed revision: empty window, cursor echoed
    resp = client.post(
        f"/sessions/{sid}/changes",
        json={
            "after_revision": 2,
            "subscriptions": [{"subscriber_id": "s", "include": ["/**"]}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["changes"] == []
    assert resp.json()["next_cursor"] == 2


# ---------------------------------------------------------------------------
# durability across restarts
# ---------------------------------------------------------------------------


def test_sealed_state_survives_store_restart(client, store):
    sid = create_session(client, {"n": 0})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", 1)])
    seal(client, sid, 1)

    app.state.session_store = SessionStore(str(store._dir))
    client = TestClient(app)
    state = client.get(f"/sessions/{sid}").json()
    assert state["sealed"] is True
    assert state["revision"] == 1
    assert state["document"] == {"n": 1}

    resp = submit(
        client, sid, 1, [op("bob", 1, {"bob": 1}, "/n", "set", 2)]
    )
    assert resp.status_code == 409
    assert "sealed" in resp.json()["detail"]
    assert replace(client, sid, 1, "zoe", "$.n", 3).status_code == 409
    # the seal is stable: a fresh process re-sealing is an idempotent 200
    assert seal(client, sid, 1).status_code == 200


# ---------------------------------------------------------------------------
# records written before the feature existed
# ---------------------------------------------------------------------------


def _write_legacy_record(store, *, revision, document, version, log):
    """Persist a record in the pre-seal on-disk format.

    Neither ``history`` nor ``sealed`` is written, exactly like a file
    left behind by an older service version.
    """
    session_id = uuid.uuid4().hex
    payload = {
        "session_id": session_id,
        "revision": revision,
        "document": document,
        "version": version,
        "log": [o.model_dump(mode="json") for o in log],
    }
    store._dir.mkdir(parents=True, exist_ok=True)
    store._path(session_id).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return session_id


def test_legacy_session_defaults_to_unsealed(store):
    sid = _write_legacy_record(
        store,
        revision=1,
        document={"x": 0},
        version={},
        log=[Operation.model_validate(op("alice", 1, {"alice": 1}, "/x", "set", 1))],
    )
    raw = json.loads(store._path(sid).read_text(encoding="utf-8"))
    assert "sealed" not in raw

    state = SessionStore(str(store._dir)).get_state(sid)
    assert state["sealed"] is False
    # still writable: the absence of the key behaves exactly as "not sealed"
    outcome = SessionStore(str(store._dir)).submit_batch(
        sid,
        1,
        [Operation.model_validate(op("bob", 1, {"bob": 1}, "/y", "set", 2))],
    )
    assert outcome.revision == 2


def test_sealing_a_legacy_session_persists_flag_and_origin_together(store, client):
    sid = _write_legacy_record(
        store,
        revision=1,
        document={"x": 0},
        version={},
        log=[Operation.model_validate(op("alice", 1, {"alice": 1}, "/x", "set", 1))],
    )
    resp = seal(client, sid, 1)
    assert resp.status_code == 200
    assert resp.json()["sealed"] is True
    raw = json.loads(store._path(sid).read_text(encoding="utf-8"))
    assert raw["sealed"] is True
    assert raw["revision"] == 1  # still no revision bump
    # the legacy history origin was seeded in the same atomic write
    assert [b["revision"] for b in raw["history"]] == [1]
    # existing-revision paging works from the seeded origin
    page = client.post(
        f"/sessions/{sid}/changes",
        json={
            "after_revision": 1,
            "subscriptions": [{"subscriber_id": "s", "include": ["/**"]}],
        },
    )
    assert page.status_code == 200
    assert page.json()["changes"] == []


# ---------------------------------------------------------------------------
# two stores/processes sharing one directory
# ---------------------------------------------------------------------------


def test_seal_in_one_store_blocks_writes_in_another(tmp_path):
    store_dir = str(tmp_path / "sessions")
    store_a = SessionStore(store_dir)
    store_b = SessionStore(store_dir)
    rec = store_a.create({"n": 0}, {})

    state = store_a.seal(rec.session_id, 0)
    assert state["sealed"] is True

    # the other "process" observes the seal straight from disk
    with pytest.raises(SessionSealedError):
        store_b.submit_batch(
            rec.session_id,
            0,
            [Operation(
                actor="bob", seq=1, version={"bob": 1},
                path="/n", action="set", value=1,
            )],
        )
    with pytest.raises(SessionSealedError):
        store_b.replace_paths(rec.session_id, 0, "bob", "$.n", 1)
    # and still reads the sealed state
    assert store_b.get_state(rec.session_id)["sealed"] is True


def test_commit_in_other_store_makes_stale_seal_fail(tmp_path):
    store_dir = str(tmp_path / "sessions")
    store_a = SessionStore(store_dir)
    store_b = SessionStore(store_dir)
    rec = store_a.create({"n": 0}, {})
    # both processes believe the session is at revision 0
    assert store_b.get(rec.session_id).revision == 0

    store_a.submit_batch(
        rec.session_id,
        0,
        [Operation(
            actor="alice", seq=1, version={"alice": 1},
            path="/n", action="set", value=1,
        )],
    )
    # sealing with the old revision loses the race: 409, stays unsealed
    with pytest.raises(RevisionConflictError):
        store_b.seal(rec.session_id, 0)
    assert store_b.get(rec.session_id).sealed is False
    # sealing at the now-current revision still succeeds
    assert store_b.seal(rec.session_id, 1)["sealed"] is True


def _racing_worker(kind, store_dir, session_id, barrier, queue):
    """Seal or commit at revision 0 after both processes pass the barrier."""
    store = SessionStore(store_dir)
    barrier.wait(timeout=15)
    try:
        if kind == "seal":
            store.seal(session_id, 0)
        else:
            store.submit_batch(
                session_id,
                0,
                [Operation(
                    actor="alice", seq=1, version={"alice": 1},
                    path="/n", action="set", value=1,
                )],
            )
        status = "ok"
    except RevisionConflictError:
        status = "conflict"
    state = store.get_state(session_id)
    queue.put((kind, status, state["revision"], state["sealed"], state["document"]))


def test_real_processes_seal_vs_commit_only_one_wins(tmp_path):
    store_dir = str(tmp_path / "sessions")
    rec = SessionStore(store_dir).create({"n": 0}, {})

    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_racing_worker,
            args=(kind, store_dir, rec.session_id, barrier, queue),
        )
        for kind in ("seal", "commit")
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0

    results = {kind: (status, rev, sealed, doc)
               for kind, status, rev, sealed, doc in
               (queue.get(timeout=10) for _ in procs)}
    seal_result = results["seal"]
    commit_result = results["commit"]

    # exactly one side succeeds; the other observes its failure under lock
    assert sorted(r[0] for r in results.values()) == ["conflict", "ok"]
    final = SessionStore(store_dir).get_state(rec.session_id)
    if seal_result[0] == "ok":
        # seal landed first: revision stays 0, document untouched,
        # the racing commit lost with 409
        assert commit_result[0] == "conflict"
        assert final["revision"] == 0
        assert final["sealed"] is True
        assert final["document"] == {"n": 0}
    else:
        # commit landed first: revision 1 unsealed, the racing seal lost
        assert commit_result[0] == "ok"
        assert final["revision"] == 1
        assert final["sealed"] is False
        assert final["document"] == {"n": 1}


def test_concurrent_seals_only_one_persists(tmp_path):
    store_dir = str(tmp_path / "sessions")
    store = SessionStore(store_dir)
    rec = store.create({}, {})

    def attempt(_):
        try:
            return SessionStore(store_dir).seal(rec.session_id, 0)
        except RevisionConflictError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(16)))
    # all share the same (correct) revision, so every seal is accepted
    # idempotently; what matters is one coherent durable result
    assert all(o is not None and o["sealed"] for o in outcomes)
    assert store.get(rec.session_id).sealed is True
    assert store.get(rec.session_id).revision == 0
