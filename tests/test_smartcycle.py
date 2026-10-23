"""Fast tests that don't need a trained model or downloaded data. Run: pytest"""

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import rules
from src.data import split
from src.labels import AXES, index_of
from src.pipeline import MAX_ATTEMPTS, Session


class FakePredictor:
    """Returns queued label sets in order, as if a trained model predicted them."""

    def __init__(self, *label_sets, trained=("material", "condition", "component"), confidence=0.9):
        self.queue = list(label_sets)
        self.trained_axes = list(trained)
        self.confidence = confidence

    def predict(self, image):
        labels = self.queue.pop(0)
        return {
            axis: {"label": labels[axis], "confidence": self.confidence, "probs": {},
                   "trained": axis in self.trained_axes}
            for axis in AXES
        }


def item(material="plastic", condition="clean", component="none"):
    return {"material": material, "condition": condition, "component": component}


def photo():
    return Image.new("RGB", (64, 64), "white")


# --- labels / rules ---------------------------------------------------------

def test_index_of_blank_and_unknown():
    assert index_of("material", "") == -1
    assert index_of("material", float("nan")) == -1
    assert index_of("material", "glass") == AXES["material"].index("glass")
    with pytest.raises(ValueError):
        index_of("material", "styrofoam")


def test_clean_item_goes_straight_to_bin():
    d = rules.decide("glass", "clean", "none")
    assert d.ready and d.bin == rules.RECYCLING


def test_dirty_item_with_lid_needs_two_steps():
    d = rules.decide("plastic", "food_residue", "lid")
    assert not d.ready and len(d.steps) == 2


def test_unfixable_condition_is_final():
    d = rules.decide("paper_cardboard", "grease", "none")
    assert d.ready and d.bin == rules.COMPOST


def test_organic_always_compost():
    assert rules.decide("organic", "liquid", "straw").bin == rules.COMPOST


def test_every_label_combination_has_a_decision():
    for m in AXES["material"]:
        for c in AXES["condition"]:
            for p in AXES["component"]:
                d = rules.decide(m, c, p)
                assert d.ready == (d.bin is not None)
                assert d.ready or d.steps


# --- data split -------------------------------------------------------------

def test_split_keeps_pairs_together_and_holds_out_source():
    rows = []
    for source in ["a", "b"]:
        for k in range(20):
            for stage in ["before", "after"]:
                rows.append({"source": source, "pair_id": str(k), "filename": f"{k}_{stage}.jpg"})
    df = pd.DataFrame(rows)
    parts = split(df, seed=1)
    for name, other in [("train", "test"), ("train", "val"), ("val", "test")]:
        assert not set(parts[name]["group"]) & set(parts[other]["group"])
    held = split(df, holdout_source="b")
    assert set(held["test"]["source"]) == {"b"}
    assert "b" not in set(held["train"]["source"]) | set(held["val"]["source"])


# --- pipeline loop ----------------------------------------------------------

def test_loop_prep_then_verified():
    s = Session(FakePredictor(item(condition="liquid"), item()))
    first = s.scan(photo())
    assert first.status == "prep" and not s.done
    second = s.scan(photo())
    assert second.status == "final" and second.decision["bin"] == rules.RECYCLING
    assert any("Verified" in m for m in second.verification)


def test_loop_reports_remaining_steps_and_material_change():
    s = Session(FakePredictor(item(condition="liquid", component="lid"), item(material="glass", component="lid")))
    s.scan(photo())
    second = s.scan(photo())
    assert second.status == "prep"
    assert any(m.startswith("Done:") for m in second.verification)
    assert any(m.startswith("Still needed:") for m in second.verification)
    assert any("same item" in m for m in second.verification)


def test_loop_gives_up_after_max_attempts():
    s = Session(FakePredictor(*[item(condition="food_residue")] * MAX_ATTEMPTS))
    for _ in range(MAX_ATTEMPTS):
        result = s.scan(photo())
    assert result.status == "final" and result.decision["bin"] == rules.TRASH


def test_untrained_axes_and_low_confidence_ask_the_user():
    s = Session(FakePredictor(item(), trained=("material",)))
    r = s.scan(photo())
    assert r.status == "need_input" and set(r.ask) == {"condition", "component"}
    r = s.answer({**r.labels, "condition": "clean", "component": "none"})
    assert r.status == "final"

    low = Session(FakePredictor(item(), confidence=0.3)).scan(photo())
    assert set(low.ask) == set(AXES)


def test_no_model_asks_everything():
    r = Session(None).scan(photo())
    assert set(r.ask) == set(AXES)


# --- web app ----------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    import app.server as server

    monkeypatch.setattr(server, "predictor", FakePredictor(item(condition="liquid"), item()))
    server.sessions.clear()
    return TestClient(server.app)


def jpeg_bytes():
    buf = io.BytesIO()
    photo().save(buf, "JPEG")
    return buf.getvalue()


def test_web_flow_shared_between_devices(client):
    sid = client.post("/api/sessions").json()["id"]
    qr = client.get(f"/api/sessions/{sid}/qr.svg")
    assert qr.status_code == 200 and qr.headers["X-Phone-Url"].endswith(f"/s/{sid}")
    assert "localhost" not in qr.headers["X-Phone-Url"] and "testserver" in qr.headers["X-Phone-Url"]

    assert client.get(f"/api/sessions/{sid}?device=phone").json()["phone_connected"]

    # phone uploads; laptop sees the same state
    client.post(f"/api/sessions/{sid}/photo", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["steps"][-1]["status"] == "prep"

    client.post(f"/api/sessions/{sid}/photo", files={"file": ("b.jpg", jpeg_bytes(), "image/jpeg")})
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["done"] and state["steps"][-1]["decision"]["bin"] == rules.RECYCLING
    assert client.get(f"/api/sessions/{sid}/photos/1").headers["content-type"] == "image/jpeg"

    assert client.post(f"/api/sessions/{sid}/photo", files={"file": ("c.jpg", jpeg_bytes(), "image/jpeg")}).status_code == 409
    assert client.post(f"/api/sessions/{sid}/reset").json()["steps"] == []


def test_web_rejects_non_images_and_unknown_sessions(client):
    sid = client.post("/api/sessions").json()["id"]
    r = client.post(f"/api/sessions/{sid}/photo", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert client.get("/api/sessions/nope").status_code == 404


def test_web_answer_flow(client, monkeypatch):
    import app.server as server

    monkeypatch.setattr(server, "predictor", None)
    sid = client.post("/api/sessions").json()["id"]
    state = client.post(f"/api/sessions/{sid}/photo", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")}).json()
    assert set(state["pending"]["ask"]) == set(AXES)
    state = client.post(f"/api/sessions/{sid}/answer", json={"answers": item(material="metal")}).json()
    assert state["pending"] is None and state["done"]
    assert client.post(f"/api/sessions/{sid}/answer", json={"answers": {}}).status_code == 409
