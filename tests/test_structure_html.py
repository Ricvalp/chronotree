"""The offline renderer must preserve data and safely contain diagnostic text."""

from html.parser import HTMLParser
import builtins
import json

import pytest

from stmem.visualization.structure_html import write_structure_html


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.external_assets = []
        self.json_text = ""
        self.in_json = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.scripts.append(attrs)
            self.in_json = attrs.get("id") == "structure-data"
        if tag in ("script", "link", "img", "iframe"):
            path = attrs.get("src", attrs.get("href"))
            if path:
                self.external_assets.append(path)

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_json = False

    def handle_data(self, data):
        if self.in_json:
            self.json_text += data


def payload():
    # Empty history and current content are legitimate; serialization never fills them.
    return {
        "schema_version": 1,
        "config": {"tau_x": 0.01, "tau_t": 4.0},
        "episode": {"task": "</title><script>unexpected()</script>"},
        "blocks": [],
        "archive": {"nodes": [], "roots": [], "leaf_ids": []},
        "current": {"leaf_ids": []},
        "provenance": {"note": "</script><script>unexpected()</script> & < > \u2028 \u2029"},
    }


def test_structure_html_is_self_contained_and_json_cannot_end_its_script(tmp_path):
    pytest.importorskip("plotly")
    record = payload()
    output = tmp_path / "nested" / "inspect.html"
    assert write_structure_html(record, output) == output
    document = Document()
    document.feed(output.read_text())
    assert len(document.scripts) == 3  # one inert JSON payload, bundled Plotly, viewer
    assert not document.external_assets
    assert json.loads(document.json_text) == record
    assert "</script>" not in document.json_text
    assert "\\u003c/script\\u003e" in document.json_text
    assert "\\u2028" in document.json_text and "\\u2029" in document.json_text


def test_structure_html_rejects_unsupported_schema_missing_fields_and_nonfinite_data(tmp_path):
    record = payload()
    record["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        write_structure_html(record, tmp_path / "unsupported.html")
    record = payload()
    del record["archive"]
    with pytest.raises(ValueError, match="missing archive"):
        write_structure_html(record, tmp_path / "missing.html")
    record = payload()
    record["config"]["tau_x"] = float("nan")
    with pytest.raises(ValueError, match="Out of range"):
        write_structure_html(record, tmp_path / "nonfinite.html")
    assert not list(tmp_path.glob("*.html"))


def test_structure_html_explains_the_optional_dependency_without_writing_partial_html(tmp_path, monkeypatch):
    original = builtins.__import__

    def without_plotly(name, *args, **kwargs):
        if name == "plotly.offline":
            raise ImportError("diagnostic: visualization extra not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_plotly)
    with pytest.raises(RuntimeError, match="--extra visualization"):
        write_structure_html(payload(), tmp_path / "missing_extra.html")
    assert not (tmp_path / "missing_extra.html").exists()
