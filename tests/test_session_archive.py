"""End-to-end tests for session archiving (finalization)."""
from __future__ import annotations

import json
import multiprocessing

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Operation
from app.sessions import (
    RevisionConflictError,
    SessionArchivedError,
    SessionStore,
)


@pytest.fixture()
def client(tmp_path):
    app.state.session_store = SessionStore(str(tmp_path / "sessions"))
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


def archive(client, session_id, expected_revision):
    return client.post(
        f"/sessions/{session_id}/archive",
        json={"expected_revision": expected_revision},
    )


# ---------------------------------------------------------------------------
# the archive response
# ---------------------------------------------------------------------------


def test_archive_returns_current_merge_state_without_revision_bump(client):
    sid = create_session(client, {"title": "draft", "keep": 1})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/title", "set", "a")])
    submit(
        client,
        sid,
        1,
        [op("bob", 1, {"bob": 1}, "/title", "set", "b")],  # concurrent, wins
    )
    submit(client, sid, 2, [op("zoe", 1, {"zoe": 1}, "/title", "delete")])

    resp = archive(client, sid, 3)
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "session_id": sid,
        "revision": 3,
        "archived": True,
        "document": {"keep": 1},
        "version": {"alice": 1, "bob": 1, "zoe": 1},
        "conflicts": body["conflicts"],
        "tombstones": ["/title"],
    }
    # the live conflict of the concurrent writes is still reported
    assert {c["path"] for c in body["conflicts"]} == {"/title"}


def test_archiving_at_revision_zero_is_allowed(client):
    sid = create_session(client, {"a": 1}, {"carol": 2})
    resp = archive(client, sid, 0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["revision"] == 0
    assert body["archived"] is True
    assert body["document"] == {"a": 1}
    assert body["version"] == {"carol": 2}
    assert body["conflicts"] == []
    assert body["tombstones"] == []


def test_archive_unknown_session_returns_404(client):
    missing = "0" * 32
    resp = archive(client, missing, 0)
    assert resp.status_code == 404
    assert client.get(f"/sessions/{missing}").status_code == 404


def test_archive_malformed_session_id_returns_404(client):
    resp = archive(client, "not-a-session-id", 0)
    assert resp.status_code == 404


def test_archive_stale_revision_returns_409_without_changing_anything(client):
    sid = create_session(client, {})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    resp = archive(client, sid, 0)
    assert resp.status_code == 409
    assert "expected revision 0" in resp.json()["detail"]
    # not archived: another write still works
    follow = submit(client, sid, 1, [op("bob", 1, {"bob": 1}, "/y", "set", 2)])
    assert follow.status_code == 200
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 2
    assert state["archived"] is False


def test_archive_future_revision_returns_409(client):
    sid = create_session(client, {})
    assert archive(client, sid, 5).status_code == 409
    assert client.get(f"/sessions/{sid}").json()["archived"] is False


# ---------------------------------------------------------------------------
# reads after archiving
# ---------------------------------------------------------------------------


def test_session_read_shows_archived_and_frozen_state(client):
    sid = create_session(client, {"n": 0})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", 1)])
    archive(client, sid, 1)

    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["archived"] is True
    assert state["document"] == {"n": 1}
    assert state["version"] == {"alice": 1}
    assert state["tombstones"] == []
    assert state["protected_paths"] == []


def test_unarchived_session_read_reports_archived_false(client):
    sid = create_session(client, {"n": 0})
    state = client.get(f"/sessions/{sid}").json()
    assert state["archived"] is False
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", 1)])
    assert client.get(f"/sessions/{sid}").json()["archived"] is False


def test_revision_changes_paging_still_works_after_archive(client):
    sid = create_session(client, {"n": 0})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", 1)])
    submit(client, sid, 1, [op("bob", 1, {"bob": 1}, "/m", "set", 2)])
    archive(client, sid, 2)

    # archiving added no event revision: paging ends at revision 2
    resp = client.post(
        f"/sessions/{sid}/changes",
        json={
            "after_revision": 0,
            "subscriptions": [{"subscriber_id": "s", "include": ["/**"]}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [c["revision"] for c in body["changes"]] == [1, 2]
    assert body["next_cursor"] == 2
    assert body["has_more"] is False

    tail = client.post(
        f"/sessions/{sid}/changes",
        json={
            "after_revision": 2,
            "subscriptions": [{"subscriber_id": "s", "include": ["/**"]}],
        },
    ).json()
    assert tail["changes"] == []
    assert tail["next_cursor"] == 2


# ---------------------------------------------------------------------------
# writes after archiving -> 409, always, without mutation
# ---------------------------------------------------------------------------


def test_batch_after_archive_returns_409(client):
    sid = create_session(client, {})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    archive(client, sid, 1)
    resp = submit(client, sid, 1, [op("bob", 1, {"bob": 1}, "/y", "set", 2)])
    assert resp.status_code == 409
    assert "archived" in resp.json()["detail"]


def test_retransmitted_batch_after_archive_returns_409(client):
    sid = create_session(client, {})
    batch = [
        op("alice", 1, {"alice": 1}, "/x", "set", 1),
        op("alice", 2, {"alice": 2}, "/y", "set", 2),
    ]
    submit(client, sid, 0, batch)
    archive(client, sid, 1)
    # exact retransmission at the archived revision: still refused
    resp = submit(client, sid, 1, batch)
    assert resp.status_code == 409
    state = client.get(f"/sessions/{sid}").json()
    assert state["revision"] == 1
    assert state["document"] == {"x": 1, "y": 2}


def test_replace_after_archive_returns_409_even_for_zero_hit(client):
    sid = create_session(client, {"x": 1})
    archive(client, sid, 0)
    resp = client.post(
        f"/sessions/{sid}/replace",
        json={
            "expected_revision": 0,
            "actor": "alice",
            "path": "$.missing",
            "value": None,
        },
    )
    assert resp.status_code == 409
    assert "archived" in resp.json()["detail"]

    # a matching replacement is refused as well
    hit = client.post(
        f"/sessions/{sid}/replace",
        json={"expected_revision": 0, "actor": "bob", "path": "$.x", "value": 2},
    )
    assert hit.status_code == 409
    state = client.get(f"/sessions/{sid}").json()
    assert state["document"] == {"x": 1}


def test_invalid_batch_after_archive_still_409_not_422(client):
    sid = create_session(client, {})
    archive(client, sid, 0)
    # alice:3 without alice:2 would ordinarily be 422; the seal wins
    resp = submit(client, sid, 0, [op("alice", 3, {"alice": 3}, "/y", "set", 2)])
    assert resp.status_code == 409


def test_archive_does_not_change_document_log_revision_or_history(client):
    sid = create_session(client, {"n": 0})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", 1)])
    before = client.get(f"/sessions/{sid}").json()
    archive(client, sid, 1)
    after = client.get(f"/sessions/{sid}").json()
    assert after["revision"] == before["revision"] == 1
    assert after["document"] == before["document"]
    assert after["version"] == before["version"]
    assert after["conflicts"] == before["conflicts"]
    assert after["tombstones"] == before["tombstones"]


# ---------------------------------------------------------------------------
# idempotent re-seal
# ---------------------------------------------------------------------------


def test_repeated_archive_is_idempotent(client):
    sid = create_session(client, {"n": 0})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", 1)])
    first = archive(client, sid, 1).json()
    second = archive(client, sid, 1).json()
    assert first == second
    assert second["archived"] is True
    assert second["revision"] == 1
    # a re-seal naming a stale revision still conflicts
    assert archive(client, sid, 0).status_code == 409


# ---------------------------------------------------------------------------
# durability across restart
# ---------------------------------------------------------------------------


def test_archive_survives_store_restart(tmp_path):
    store_dir = str(tmp_path / "sessions")
    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    sid = create_session(client, {"n": 0})
    submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/n", "set", 1)])
    archive(client, sid, 1)

    app.state.session_store = SessionStore(store_dir)
    client = TestClient(app)
    state = client.get(f"/sessions/{sid}").json()
    assert state["archived"] is True
    assert state["revision"] == 1
    assert state["document"] == {"n": 1}

    # writes stay refused after the restart
    assert submit(client, sid, 1, [op("bob", 1, {"bob": 1}, "/m", "set", 2)]).status_code == 409
    resp = client.post(
        f"/sessions/{sid}/replace",
        json={"expected_revision": 1, "actor": "bob", "path": "$.n", "value": 9},
    )
    assert resp.status_code == 409

    # re-sealing after the restart returns the same frozen result
    again = archive(client, sid, 1).json()
    assert again["document"] == {"n": 1}
    assert again["revision"] == 1


def test_legacy_session_without_archive_field_is_open_by_default(tmp_path):
    # hand-write a record the way an older service version would (no
    # history, no archived field) and confirm it reads as un-archived
    store_dir = tmp_path / "sessions"
    store_dir.mkdir()
    sid = "a" * 32
    (store_dir / f"{sid}.json").write_text(
        json.dumps(
            {
                "session_id": sid,
                "revision": 0,
                "document": {"old": True},
                "version": {},
                "log": [],
            }
        )
    )
    app.state.session_store = SessionStore(str(store_dir))
    client = TestClient(app)
    state = client.get(f"/sessions/{sid}").json()
    assert state["archived"] is False
    # it still accepts writes as before
    resp = submit(client, sid, 0, [op("alice", 1, {"alice": 1}, "/x", "set", 1)])
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# store-level API
# ---------------------------------------------------------------------------


def test_store_archive_then_write_raises_session_archived_error(tmp_path):
    store = SessionStore(str(tmp_path / "sessions"))
    rec = store.create({}, {})
    result = store.archive_session(rec.session_id, 0)
    assert result["archived"] is True
    assert result["revision"] == 0
    with pytest.raises(SessionArchivedError):
        store.submit_batch(
            rec.session_id,
            0,
            [Operation(actor="a", seq=1, version={"a": 1}, path="/x", action="set", value=1)],
        )
    # SessionArchivedError is a RevisionConflictError as well
    with pytest.raises(RevisionConflictError):
        store.archive_session(rec.session_id, 3)
    assert store.get(rec.session_id).archived is True


# ---------------------------------------------------------------------------
# two real processes sharing one store directory
# ---------------------------------------------------------------------------


def _archive_worker(store_dir, session_id, expected, queue):
    store = SessionStore(store_dir)
    try:
        store.archive_session(session_id, expected)
        status = "archived"
    except RevisionConflictError:
        status = "conflict"
    state = store.get_state(session_id)
    queue.put(
        (
            status,
            state["revision"],
            state["archived"],
            state["document"],
        )
    )


def _submit_worker(store_dir, session_id, expected, queue):
    store = SessionStore(store_dir)
    operation = Operation(
        actor="carol",
        seq=1,
        version={"carol": 1},
        path="/late",
        action="set",
        value="late",
    )
    try:
        store.submit_batch(session_id, expected, [operation])
        status = "committed"
    except RevisionConflictError:
        status = "conflict"
    state = store.get_state(session_id)
    queue.put(
        (
            status,
            state["revision"],
            state["archived"],
            state["document"],
        )
    )


def test_archive_and_commit_race_across_processes_only_one_wins(tmp_path):
    # both processes start from revision 1; one seals, one commits.  The
    # loser must see 409 and the durable end state must be self-consistent.
    store_dir = str(tmp_path / "sessions")
    store = SessionStore(store_dir)
    rec = store.create({"n": 0}, {})
    winner_op = Operation(
        actor="alice", seq=1, version={"alice": 1}, path="/n", action="set", value=1
    )
    store.submit_batch(rec.session_id, 0, [winner_op])

    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)

    def barrier_archive(store_dir, session_id, expected, queue):
        SessionStore(store_dir).get(session_id)
        barrier.wait(timeout=15)
        _archive_worker(store_dir, session_id, expected, queue)

    def barrier_submit(store_dir, session_id, expected, queue):
        SessionStore(store_dir).get(session_id)
        barrier.wait(timeout=15)
        _submit_worker(store_dir, session_id, expected, queue)

    queue = ctx.Queue()
    procs = [
        ctx.Process(target=barrier_archive, args=(store_dir, rec.session_id, 1, queue)),
        ctx.Process(target=barrier_submit, args=(store_dir, rec.session_id, 1, queue)),
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0

    results = [queue.get(timeout=10) for _ in procs]
    statuses = sorted(r[0] for r in results)
    # exactly one side wins; the loser observes a 409
    assert statuses in (
        ["archived", "conflict"],
        ["committed", "conflict"],
    )

    final = SessionStore(store_dir).get_state(rec.session_id)
    if statuses == ["archived", "conflict"]:
        # archive won: revision stays 1, no /late key, sealed
        assert final["revision"] == 1
        assert final["archived"] is True
        assert final["document"] == {"n": 1}
        # and now even a fresh process's commit is refused
        queue2 = ctx.Queue()
        p = ctx.Process(
            target=_submit_worker, args=(store_dir, rec.session_id, 1, queue2)
        )
        p.start()
        p.join(timeout=60)
        assert p.exitcode == 0
        status, revision, archived, document = queue2.get(timeout=10)
        assert status == "conflict"
        assert revision == 1
        assert archived is True
        assert document == {"n": 1}
    else:
        # commit won first: revision moved to 2, archive's stale expected=1
        # conflicts, session stays open and is not sealed
        assert final["revision"] == 2
        assert final["archived"] is False
        assert final["document"] == {"n": 1, "late": "late"}
        # but archiving at the new revision still succeeds afterwards
        queue2 = ctx.Queue()
        p = ctx.Process(
            target=_archive_worker, args=(store_dir, rec.session_id, 2, queue2)
        )
        p.start()
        p.join(timeout=60)
        assert p.exitcode == 0
        status, revision, archived, document = queue2.get(timeout=10)
        assert status == "archived"
        assert revision == 2
        assert archived is True
        assert document == {"n": 1, "late": "late"}
