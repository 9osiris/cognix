"""Schemas: frame-based interpretation of parsed observations.

A Schema is a named frame with slots. The SchemaLibrary matches parsed
observations against registered schemas and fills slots from entities,
numbers, relations and keywords.
"""

import re

# where a slot value can come from inside a parsed observation
SLOT_SOURCES = (
    "entity",
    "number",
    "relation_subject",
    "relation_object",
    "relation_verb",
    "keyword",
    "dateword",
    "text",
)

_DATEWORD_RE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|january|february|march|april|may|june|july|"
    r"august|september|october|november|december)\b",
    re.IGNORECASE,
)


def _normalize_slot(spec):
    """Accept a description string, a (description, required) pair, or a dict."""
    if isinstance(spec, str):
        spec = {"description": spec}
    elif isinstance(spec, (tuple, list)):
        spec = {"description": spec[0], "required": bool(spec[1]) if len(spec) > 1 else False}
    else:
        spec = dict(spec)
    normalized = {
        "description": str(spec.get("description", "")),
        "required": bool(spec.get("required", False)),
        "source": spec.get("source", "entity"),
    }
    for key in ("predicate", "index", "match"):
        if key in spec:
            normalized[key] = spec[key]
    if normalized["source"] not in SLOT_SOURCES:
        raise ValueError("unknown slot source: %r" % (normalized["source"],))
    return normalized


def _find_relation(relations, predicates):
    if not relations:
        return None
    if predicates:
        wanted = {str(item).lower() for item in predicates}
        for relation in relations:
            if str(relation[1]).lower() in wanted:
                return relation
        return None
    return relations[0]


def _find_dateword(text):
    match = _DATEWORD_RE.search(text or "")
    return match.group(1).lower() if match else None


def _extract_slot(spec, parsed):
    """Best-effort value for one slot spec, or None if unavailable."""
    source = spec.get("source", "entity")
    index = spec.get("index", 0)
    if source == "entity":
        items = parsed.get("entities", [])
        return items[index] if 0 <= index < len(items) else None
    if source == "number":
        items = parsed.get("numbers", [])
        return items[index]["value"] if 0 <= index < len(items) else None
    if source in ("relation_subject", "relation_object", "relation_verb"):
        relation = _find_relation(parsed.get("relations", []), spec.get("predicate"))
        if relation is None:
            return None
        return {
            "relation_subject": relation[0],
            "relation_object": relation[2],
            "relation_verb": relation[1],
        }[source]
    if source == "keyword":
        items = parsed.get("keywords", [])
        return items[index] if 0 <= index < len(items) else None
    if source == "dateword":
        return _find_dateword(parsed.get("text", ""))
    if source == "text":
        return parsed.get("text") or None
    return None


class Schema:
    """A named frame: slots describe what to look for, fillers hold values."""

    def __init__(self, name, slots=None, fillers=None):
        self.name = name
        self.slots = {
            slot: _normalize_slot(spec) for slot, spec in (slots or {}).items()
        }
        self.fillers = dict(fillers or {})

    def __repr__(self):
        return "Schema(%r, %d slots)" % (self.name, len(self.slots))

    def fill_slot(self, slot, value):
        if slot not in self.slots:
            raise KeyError("unknown slot %r for schema %r" % (slot, self.name))
        self.fillers[slot] = value
        return value

    def missing(self):
        """Required slots that have no filler yet."""
        return [
            slot
            for slot, spec in self.slots.items()
            if spec["required"] and slot not in self.fillers
        ]

    def is_complete(self):
        return not self.missing()

    def describe(self):
        """One-line-per-slot human summary of the schema."""
        lines = ["schema %s:" % self.name]
        for slot, spec in self.slots.items():
            kind = "required" if spec["required"] else "optional"
            lines.append("  %s (%s): %s" % (slot, kind, spec["description"]))
        return "\n".join(lines)

    def fillable_count(self, parsed):
        """How many slots can be filled from this parsed observation."""
        return sum(
            1 for spec in self.slots.values() if _extract_slot(spec, parsed) is not None
        )

    def to_dict(self):
        return {
            "name": self.name,
            "slots": self.slots,
            "fillers": dict(self.fillers),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], slots=data.get("slots", {}), fillers=data.get("fillers", {}))


class SchemaLibrary:
    """Registry of schemas with match and fill against parsed observations."""

    def __init__(self):
        self._schemas = {}
        self._register_defaults()

    def _register_defaults(self):
        self.define(
            "person",
            {
                "name": {
                    "description": "the person's name",
                    "required": True,
                    "source": "entity",
                },
                "age": {
                    "description": "the person's age in years",
                    "required": False,
                    "source": "number",
                },
                "occupation": {
                    "description": "what the person does",
                    "required": False,
                    "source": "relation_object",
                    "predicate": ["is", "are", "was"],
                },
            },
        )
        self.define(
            "object_fact",
            {
                "subject": {
                    "description": "the thing being described",
                    "required": True,
                    "source": "relation_subject",
                    "predicate": ["is", "are", "was", "were"],
                },
                "attribute": {
                    "description": "what is said about the subject",
                    "required": True,
                    "source": "relation_object",
                    "predicate": ["is", "are", "was", "were"],
                },
            },
        )
        self.define(
            "preference",
            {
                "subject": {
                    "description": "who holds the preference",
                    "required": True,
                    "source": "relation_subject",
                    "predicate": ["likes", "like", "loves", "love", "wants", "want"],
                },
                "liked": {
                    "description": "the thing that is preferred",
                    "required": True,
                    "source": "relation_object",
                    "predicate": ["likes", "like", "loves", "love", "wants", "want"],
                },
            },
        )
        self.define(
            "event",
            {
                "who": {
                    "description": "who acted",
                    "required": True,
                    "source": "entity",
                },
                "action": {
                    "description": "what was done",
                    "required": True,
                    "source": "relation_verb",
                },
                "when": {
                    "description": "when it happened",
                    "required": False,
                    "source": "dateword",
                },
            },
        )

    def register(self, schema):
        self._schemas[schema.name] = schema
        return schema

    def unregister(self, name):
        """Remove a schema, returning True if it existed."""
        return self._schemas.pop(name, None) is not None

    def define(self, name, slots):
        """Convenience: build a Schema from a slots dict and register it."""
        return self.register(Schema(name, slots))

    def get(self, name):
        return self._schemas.get(name)

    def names(self):
        return list(self._schemas)

    def match(self, parsed):
        """Score schemas by fraction of slots fillable from parsed.

        Returns [(Schema, score)] sorted best first, only scores above 0.
        """
        scored = []
        for schema in self._schemas.values():
            total = len(schema.slots)
            if not total:
                continue
            score = schema.fillable_count(parsed) / total
            if score > 0:
                scored.append((schema, score))
        scored.sort(key=lambda item: (-item[1], item[0].name))
        return scored

    def fill(self, schema_name, parsed):
        """Best-effort slot values for a schema; None where unavailable."""
        schema = self._schemas.get(schema_name)
        if schema is None:
            raise KeyError("unknown schema %r" % (schema_name,))
        return {
            slot: _extract_slot(spec, parsed)
            for slot, spec in schema.slots.items()
        }

    def fill_all(self, parsed):
        """Fill every matching schema, keyed by schema name."""
        return {
            schema.name: self.fill(schema.name, parsed)
            for schema, _ in self.match(parsed)
        }

    def to_dict(self):
        return {"schemas": [schema.to_dict() for schema in self._schemas.values()]}

    @classmethod
    def from_dict(cls, data):
        library = cls.__new__(cls)
        library._schemas = {}
        for item in data.get("schemas", []):
            schema = Schema.from_dict(item)
            library._schemas[schema.name] = schema
        return library
