"""Query parser — uses Anthropic Claude to convert natural language to structured filters."""

from __future__ import annotations

import json

import anthropic
import structlog

from src.config import settings
from src.query_api.schemas import ParsedQuery

log = structlog.get_logger()

_SYSTEM_PROMPT = (
    "You parse CCTV search queries into JSON only. No prose, no fences.\n"
    'Return exactly: {"class_filter":["car"] or null,"attribute_filters":'
    '{"color":"red"},"camera_ids":["cam_01"] or null,"time_range":'
    '{"after":"14:00","before":null} or null,"event_type":"tracking" or '
    '"scene_diff" or "any","text_query":"concise visual description","limit":10}\n'
    "event_type is scene_diff only for removal/theft/disappearance queries."
)


class QueryParser:
    """Parses natural-language CCTV queries into structured ParsedQuery objects."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def parse(self, query: str) -> ParsedQuery:
        """Send the query to Claude and parse the JSON response.

        Falls back to a default ParsedQuery(text_query=query) on failure.
        """
        try:
            message = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )

            raw_text = message.content[0].text.strip()
            data = json.loads(raw_text)
            return ParsedQuery.model_validate(data)

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            log.warning("query_parse_json_error", query=query, error=str(exc))
            return ParsedQuery(text_query=query)

        except Exception as exc:
            log.error("query_parse_failed", query=query, error=str(exc))
            return ParsedQuery(text_query=query)
