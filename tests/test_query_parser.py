"""Tests for QueryParser — SpaCy-based, no mocks needed."""

from __future__ import annotations

from src.query_api.nlp.query_parser import QueryParser


def test_parse_tracking_query():
    """Parser extracts class filter and color from natural language."""
    parser = QueryParser()
    result = parser.parse("find red car")

    assert result.class_filter is not None
    assert "car" in result.class_filter
    assert result.attribute_filters.get("color") == "red"
    assert result.text_query  # should have something


def test_parse_person_query():
    """Parser identifies people-related queries."""
    parser = QueryParser()
    result = parser.parse("show me the person in blue clothing")

    assert result.class_filter is not None
    assert "person" in result.class_filter
    assert result.attribute_filters.get("color") == "blue"


def test_parse_scene_diff_query():
    """Parser detects removal/theft as scene_diff event type."""
    parser = QueryParser()
    result = parser.parse("when was the bag removed from the table")

    assert result.event_type == "scene_diff"


def test_parse_aliases():
    """Parser resolves aliases like 'people' → 'person'."""
    parser = QueryParser()
    result = parser.parse("find people walking")

    assert result.class_filter is not None
    assert "person" in result.class_filter


def test_parse_camera_id():
    """Parser extracts camera IDs like cam_01."""
    parser = QueryParser()
    result = parser.parse("show events from cam_01")

    assert result.camera_ids is not None
    assert any("01" in c for c in result.camera_ids)


def test_parse_time_range():
    """Parser extracts time references."""
    parser = QueryParser()
    result = parser.parse("find cars after 2pm")

    assert result.time_range is not None
    assert "after" in result.time_range


def test_parse_fallback():
    """Parser returns a valid ParsedQuery even for gibberish."""
    parser = QueryParser()
    result = parser.parse("xyzzy foobar baz")

    assert result.text_query == "xyzzy foobar baz"
    assert result.event_type == "any"
