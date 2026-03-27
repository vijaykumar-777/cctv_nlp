"""Tests for QueryParser — mocks Anthropic API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.query_api.nlp.query_parser import QueryParser


def _mock_anthropic_response(json_str: str):
    """Create a mock Message object with the given text content."""
    mock_msg = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json_str
    mock_msg.content = [mock_content]
    return mock_msg


@patch("src.query_api.nlp.query_parser.anthropic.Anthropic")
def test_parse_tracking_query(mock_anthropic_cls):
    """Parser returns structured ParsedQuery from Claude's JSON response."""
    response_json = json.dumps({
        "class_filter": ["car"],
        "attribute_filters": {"color": "red"},
        "camera_ids": None,
        "time_range": None,
        "event_type": "tracking",
        "text_query": "red car",
        "limit": 10,
    })

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(response_json)
    mock_anthropic_cls.return_value = mock_client

    parser = QueryParser()
    result = parser.parse("find red car")

    assert result.class_filter == ["car"]
    assert result.attribute_filters == {"color": "red"}
    assert result.event_type == "tracking"
    assert result.text_query == "red car"


@patch("src.query_api.nlp.query_parser.anthropic.Anthropic")
def test_parse_scene_diff_query(mock_anthropic_cls):
    """Parser correctly identifies scene_diff event_type."""
    response_json = json.dumps({
        "class_filter": ["bag"],
        "attribute_filters": {},
        "camera_ids": None,
        "time_range": None,
        "event_type": "scene_diff",
        "text_query": "bag removed from table",
        "limit": 5,
    })

    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response(response_json)
    mock_anthropic_cls.return_value = mock_client

    parser = QueryParser()
    result = parser.parse("when was the bag removed from the table")

    assert result.event_type == "scene_diff"
    assert result.class_filter == ["bag"]


@patch("src.query_api.nlp.query_parser.anthropic.Anthropic")
def test_parse_invalid_json_fallback(mock_anthropic_cls):
    """Parser falls back to default ParsedQuery when Claude returns invalid JSON."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_anthropic_response("not valid json")
    mock_anthropic_cls.return_value = mock_client

    parser = QueryParser()
    result = parser.parse("some random query")

    assert result.text_query == "some random query"
    assert result.class_filter is None
    assert result.event_type == "any"
