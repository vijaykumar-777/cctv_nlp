"""Query parser — uses SpaCy NLP to convert natural language to structured filters.

Runs entirely offline — no API key needed.
"""

from __future__ import annotations

import re

import spacy
import structlog

from src.query_api.schemas import ParsedQuery

log = structlog.get_logger()

# ── COCO class names that YOLO can detect ────────────────────────
COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}

# Common aliases → canonical COCO class
CLASS_ALIASES = {
    "people": "person", "man": "person", "woman": "person",
    "pedestrian": "person", "guy": "person", "lady": "person",
    "child": "person", "kid": "person",
    "vehicle": "car", "auto": "car", "automobile": "car",
    "van": "car", "suv": "car",
    "motorbike": "motorcycle", "bike": "bicycle",
    "lorry": "truck", "rig": "truck",
    "bag": "backpack", "luggage": "suitcase",
    "phone": "cell phone", "mobile": "cell phone",
    "monitor": "tv", "screen": "tv", "television": "tv",
    "sofa": "couch", "table": "dining table",
    "plant": "potted plant", "flower": "potted plant",
}

# Colours the AttributeExtractor knows
KNOWN_COLORS = {
    "red", "orange", "yellow", "green", "cyan", "blue",
    "purple", "pink", "white", "black", "gray", "grey",
    "brown", "silver", "gold", "beige", "maroon", "navy",
}

# Scene-diff trigger words
REMOVAL_KEYWORDS = {
    "removed", "missing", "gone", "disappeared", "stolen",
    "theft", "taken", "lost", "vanished",
}

# Time patterns
TIME_RE = re.compile(
    r"(?:after|before|from|since|until|till|at)\s+"
    r"(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)",
    re.IGNORECASE,
)

CAMERA_RE = re.compile(r"cam(?:era)?[_\s-]?(\w+)", re.IGNORECASE)


class QueryParser:
    """Parses natural-language CCTV queries into structured ParsedQuery objects.

    Uses SpaCy for NER/POS tagging plus rule-based extraction for
    colours, COCO classes, time ranges, and camera IDs.
    No external API key required.
    """

    def __init__(self) -> None:
        self._nlp = spacy.load("en_core_web_sm")

    def parse(self, query: str) -> ParsedQuery:
        """Parse a free-text query into structured filters."""
        try:
            doc = self._nlp(query.lower())

            class_filter = self._extract_classes(doc)
            attribute_filters = self._extract_attributes(doc)
            camera_ids = self._extract_cameras(query)
            time_range = self._extract_time(query)
            event_type = self._detect_event_type(doc)
            text_query = self._build_text_query(query, class_filter, attribute_filters)

            return ParsedQuery(
                class_filter=class_filter or None,
                attribute_filters=attribute_filters,
                camera_ids=camera_ids or None,
                time_range=time_range,
                event_type=event_type,
                text_query=text_query,
                limit=10,
            )
        except Exception as exc:
            log.warning("spacy_parse_error", query=query, error=str(exc))
            return ParsedQuery(text_query=query)

    def _extract_classes(self, doc) -> list[str]:
        """Find COCO object classes mentioned in the query."""
        found = []
        text = doc.text

        # Check multi-word classes first (e.g. "traffic light")
        for cls in sorted(COCO_CLASSES, key=len, reverse=True):
            if cls in text:
                found.append(cls)
                text = text.replace(cls, "")

        # Check single-token aliases
        for token in doc:
            lemma = token.lemma_
            word = token.text
            for w in (word, lemma):
                if w in CLASS_ALIASES and CLASS_ALIASES[w] not in found:
                    found.append(CLASS_ALIASES[w])
                elif w in COCO_CLASSES and w not in found:
                    found.append(w)

        return found

    def _extract_attributes(self, doc) -> dict:
        """Extract colour and other attributes from the query."""
        attrs = {}
        for token in doc:
            if token.text in KNOWN_COLORS:
                # normalise grey → gray
                color = "gray" if token.text == "grey" else token.text
                attrs["color"] = color
                break
        return attrs

    def _extract_cameras(self, query: str) -> list[str]:
        """Extract camera IDs like 'cam_01' or 'camera 2'."""
        matches = CAMERA_RE.findall(query)
        return [f"cam_{m}" for m in matches] if matches else []

    def _extract_time(self, query: str) -> dict | None:
        """Extract time references (after/before HH:MM)."""
        matches = TIME_RE.findall(query.lower())
        if not matches:
            return None

        time_range: dict = {}
        lower = query.lower()

        for match in matches:
            # Check if this time is preceded by "after" or "before"
            idx = lower.find(match)
            prefix = lower[max(0, idx - 10):idx].strip()

            if any(w in prefix for w in ("after", "from", "since")):
                time_range["after"] = match.strip()
            elif any(w in prefix for w in ("before", "until", "till")):
                time_range["before"] = match.strip()
            else:
                time_range["after"] = match.strip()

        return time_range if time_range else None

    def _detect_event_type(self, doc) -> str:
        """Detect if the query is about scene changes (removal/theft)."""
        for token in doc:
            if token.text in REMOVAL_KEYWORDS or token.lemma_ in REMOVAL_KEYWORDS:
                return "scene_diff"
        return "any"

    def _build_text_query(
        self, query: str, classes: list[str], attrs: dict
    ) -> str:
        """Build a concise visual description for CLIP vector search."""
        parts = []
        if attrs.get("color"):
            parts.append(attrs["color"])
        if classes:
            parts.append(classes[0])
        if parts:
            return " ".join(parts)
        # Fallback: use the original query cleaned up
        return query.strip()
