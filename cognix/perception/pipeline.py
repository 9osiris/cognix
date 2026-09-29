"""Perception pipeline: raw text to structured observations.

parse_observation turns a raw string into entities, numbers, relations,
intent, urgency flags and keywords. observation_to_beliefs converts the
parsed form into candidate belief propositions.

This module also owns the shared text utilities (stopwords, keyword
extraction, urgency markers) used by the attention subsystem.
"""

import re

# words with no topical content, dropped from keyword lists
STOPWORDS = frozenset(
    "a an the and or but if then else when while of at by for with about into "
    "through during before after above below to from up down in out on off over "
    "under again further once here there where why how all any both each few more "
    "most other some such no nor not only own same so than too very can will just "
    "should now i me my myself we our ours you your he him his she her it its they "
    "them their what which who whom this that these those am is are was were be "
    "been being have has had having do does did doing would could ought as".split()
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def keywords(text):
    """Lowercased content words minus stopwords, order preserved, deduped."""
    words = []
    for match in _WORD_RE.finditer(text or ""):
        word = match.group(0).lower().strip("'")
        if len(word) > 1 and word not in STOPWORDS and word not in words:
            words.append(word)
    return words


# urgency cue words/phrases with weights used by salience scoring
URGENCY_MARKERS = {
    "urgent": 0.5,
    "asap": 0.5,
    "critical": 0.5,
    "emergency": 0.5,
    "immediately": 0.4,
    "deadline": 0.4,
    "as soon as possible": 0.4,
    "due": 0.3,
    "right now": 0.3,
    "!": 0.3,
    "now": 0.2,
}

_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
)
_DATE_WORDS = frozenset(
    "today tonight tomorrow yesterday monday tuesday wednesday thursday friday "
    "saturday sunday january february march april may june july august september "
    "october november december".split()
)


def find_urgency_markers(text):
    """Return the urgency markers present in text, in dict order."""
    found = []
    low = (text or "").lower()
    for marker in URGENCY_MARKERS:
        if marker == "!":
            if "!" in (text or ""):
                found.append(marker)
        elif re.search(r"(?<![a-z])" + re.escape(marker) + r"(?![a-z])", low):
            found.append(marker)
    return found


def mentions_date(text):
    """True if text mentions a calendar date, day or deadline word."""
    if _DATE_RE.search(text or ""):
        return True
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    return bool(words & _DATE_WORDS)


# capitalized word sequences, e.g. "Alice", "New York", "NASA"
_ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\b")
_ENTITY_EXCLUSIONS = frozenset(
    "i a an the this that these those it he she they we you my your his her its "
    "their our what which who whom whose when where why how".split()
)
_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _extract_entities(text):
    entities = []
    for match in _ENTITY_RE.finditer(text):
        entity = _LEADING_ARTICLE_RE.sub("", match.group(1)).strip()
        if len(entity) < 2:
            continue
        if " " not in entity and entity.lower() in _ENTITY_EXCLUSIONS:
            continue
        if entity not in entities:
            entities.append(entity)
    return entities


_NUMBER_RE = re.compile(r"(?P<num>[-+]?\d[\d,]*(?:\.\d+)?)")
_CONTEXT_WORD_RE = re.compile(r"[A-Za-z']+")


def _extract_numbers(text):
    numbers = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group("num").replace(",", "")
        value = float(raw) if "." in raw else int(raw)
        before = _CONTEXT_WORD_RE.findall(text[: match.start()])[-3:]
        after = _CONTEXT_WORD_RE.findall(text[match.end():])[:3]
        numbers.append(
            {
                "value": value,
                "raw": match.group("num"),
                "context": " ".join(before + after),
            }
        )
    return numbers


_RELATION_VERBS = (
    "is", "are", "was", "were",
    "has", "have", "had",
    "likes", "like", "loves", "love",
    "wants", "want", "needs", "need",
    "bought", "buys", "buy",
    "sold", "sells", "sell",
    "met", "meets", "meet",
    "visited", "visits", "visit",
    "called", "calls", "call",
    "knows", "knew", "know",
    "made", "makes", "make",
)
_VERB_ALT = "|".join(sorted(_RELATION_VERBS, key=len, reverse=True))
_SUBJECT = (
    r"(?P<subj>(?:[Tt]he\s+|[Aa]n?\s+)?"
    r"(?:[A-Za-z][\w']*|[Ii]|[Hh]e|[Ss]he|[Tt]hey|[Ww]e|[Yy]ou)"
    r"(?:\s+[A-Za-z][\w']*)*)"
)
_RELATION_RE = re.compile(
    _SUBJECT + r"\s+(?P<verb>" + _VERB_ALT + r")\s+(?P<obj>[^.,!?;:]+)"
)
def _extract_relations(text):
    relations = []
    for match in _RELATION_RE.finditer(text):
        subject = _LEADING_ARTICLE_RE.sub("", match.group("subj")).strip()
        verb = match.group("verb").lower()
        obj = _LEADING_ARTICLE_RE.sub("", match.group("obj")).strip()
        obj = re.sub(r"\s+", " ", obj)
        if subject and obj:
            relations.append((subject, verb, obj))
    return relations


_QUESTION_WORDS = frozenset(
    "who what when where why how which whom whose is are was were do does did "
    "can could should would will have has am".split()
)
_COMMAND_VERBS = frozenset(
    "do make get set send write run stop start open close buy call tell give take "
    "go come check find build create delete remove add fix try remember note "
    "summarize explain list show print save load fetch push pull commit deploy "
    "test".split()
)
_GOAL_CUES = (
    "i want to",
    "i need to",
    "my goal",
    "i plan to",
    "planning to",
    "going to",
    "i should",
    "we should",
    "let's ",
)


def _detect_intent(text):
    low = text.lower().strip()
    if not low:
        return "inform"
    if low.endswith("?"):
        return "question"
    if any(cue in low for cue in _GOAL_CUES):
        return "goal"
    first = re.findall(r"[a-z']+", low)
    first_word = first[0] if first else ""
    if low.startswith("please ") or first_word in _COMMAND_VERBS:
        return "command"
    if first_word in _QUESTION_WORDS:
        return "question"
    return "inform"


def _detect_urgency(text):
    return bool(find_urgency_markers(text)) or mentions_date(text)


def parse_observation(raw):
    """Parse raw text into a structured observation dict."""
    text = (raw or "").strip()
    return {
        "text": text,
        "entities": _extract_entities(text),
        "numbers": _extract_numbers(text),
        "relations": _extract_relations(text),
        "intent": _detect_intent(text),
        "urgency": _detect_urgency(text),
        "keywords": keywords(text),
    }


def observation_to_beliefs(parsed):
    """Turn a parsed observation into (proposition, confidence) pairs.

    Relations become "subject predicate object" assertions at 0.7,
    entities become "entity exists" assertions at 0.5. Questions yield
    no assertions, they are queries not beliefs.
    """
    if parsed.get("intent") == "question":
        return []
    beliefs = []
    seen = set()

    def add(proposition, confidence):
        proposition = re.sub(r"\s+", " ", proposition.lower()).strip()
        if proposition and proposition not in seen:
            seen.add(proposition)
            beliefs.append((proposition, confidence))

    for subject, verb, obj in parsed.get("relations", []):
        add("%s %s %s" % (subject, verb, obj), 0.7)
    for entity in parsed.get("entities", []):
        add("%s exists" % entity, 0.5)
    return beliefs
