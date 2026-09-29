"""Tests for the perception subsystem: pipeline, beliefs, schemas."""

import pytest

from cognix.perception import (
    STOPWORDS,
    Belief,
    BeliefStore,
    Schema,
    SchemaLibrary,
    find_urgency_markers,
    keywords,
    mentions_date,
    observation_to_beliefs,
    parse_observation,
)


class TestKeywords:
    def test_drops_stopwords_and_lowercases(self):
        assert keywords("The quick brown fox jumps") == ["quick", "brown", "fox", "jumps"]

    def test_dedupes_and_skips_single_letters(self):
        assert keywords("cat cat c dog") == ["cat", "dog"]

    def test_empty_and_none_safe(self):
        assert keywords("") == []
        assert keywords(None) == []

    def test_stopwords_is_nontrivial(self):
        assert len(STOPWORDS) > 50
        assert "the" in STOPWORDS


class TestUrgencyHelpers:
    def test_find_urgency_markers(self):
        markers = find_urgency_markers("URGENT: fix this asap!")
        assert "urgent" in markers
        assert "asap" in markers
        assert "!" in markers
        assert find_urgency_markers("nothing special") == []

    def test_mentions_date(self):
        assert mentions_date("see you on friday") is True
        assert mentions_date("due 2026-10-01") is True
        assert mentions_date("just a normal sentence") is False


class TestParseObservation:
    def test_keys_present(self):
        parsed = parse_observation("Alice likes pizza.")
        assert set(parsed) == {
            "text",
            "entities",
            "numbers",
            "relations",
            "intent",
            "urgency",
            "keywords",
        }
        assert parsed["text"] == "Alice likes pizza."

    def test_extracts_entities(self):
        parsed = parse_observation("Alice visited Paris in 2024.")
        assert "Alice" in parsed["entities"]
        assert "Paris" in parsed["entities"]
        assert "2024" not in parsed["entities"]

    def test_extracts_multiword_entities(self):
        parsed = parse_observation("The Eiffel Tower is in Paris.")
        assert "Eiffel Tower" in parsed["entities"]

    def test_sentence_start_words_are_not_entities(self):
        parsed = parse_observation("The cat sat on the mat.")
        assert parsed["entities"] == []

    def test_extracts_numbers_with_types_and_context(self):
        parsed = parse_observation("it costs 300 dollars and 4.5 euros")
        by_value = {item["value"]: item for item in parsed["numbers"]}
        assert by_value[300]["raw"] == "300"
        assert isinstance(by_value[300]["value"], int)
        assert isinstance(by_value[4.5]["value"], float)
        assert "dollars" in by_value[300]["context"]
        assert "euros" in by_value[4.5]["context"]

    def test_extracts_relations(self):
        parsed = parse_observation("Bob likes pizza.")
        assert ("Bob", "likes", "pizza") in parsed["relations"]

    def test_extracts_is_relations_with_articles_stripped(self):
        parsed = parse_observation("The Eiffel Tower is 330 meters tall.")
        assert ("Eiffel Tower", "is", "330 meters tall") in parsed["relations"]

    def test_extracts_have_relations(self):
        parsed = parse_observation("She has a cat.")
        assert ("She", "has", "cat") in parsed["relations"]

    def test_intent_question(self):
        assert parse_observation("What is the capital of France?")["intent"] == "question"
        assert parse_observation("Is the store open?")["intent"] == "question"

    def test_intent_command(self):
        assert parse_observation("Buy milk now!")["intent"] == "command"
        assert parse_observation("Please close the door.")["intent"] == "command"

    def test_intent_goal(self):
        assert parse_observation("I want to learn Spanish.")["intent"] == "goal"
        assert parse_observation("My goal is to run a marathon.")["intent"] == "goal"

    def test_intent_inform_default(self):
        assert parse_observation("The sky is blue.")["intent"] == "inform"
        assert parse_observation("")["intent"] == "inform"

    def test_urgency_flag(self):
        assert parse_observation("URGENT: fix this asap!")["urgency"] is True
        assert parse_observation("Submit the report by Friday.")["urgency"] is True
        assert parse_observation("The sky is blue.")["urgency"] is False

    def test_keywords_included(self):
        parsed = parse_observation("Alice quickly bought fresh pizza")
        assert "pizza" in parsed["keywords"]
        assert "alice" in parsed["keywords"]


class TestObservationToBeliefs:
    def test_relations_become_propositions_at_point_seven(self):
        parsed = parse_observation("Bob likes pizza.")
        beliefs = observation_to_beliefs(parsed)
        assert ("bob likes pizza", 0.7) in beliefs

    def test_entities_become_exists_at_point_five(self):
        parsed = parse_observation("Bob likes pizza.")
        beliefs = observation_to_beliefs(parsed)
        assert ("bob exists", 0.5) in beliefs
        assert ("pizza exists", 0.5) not in beliefs  # lowercase, not an entity

    def test_questions_produce_no_assertions(self):
        parsed = parse_observation("What is the capital of France?")
        assert observation_to_beliefs(parsed) == []

    def test_dedupes_propositions(self):
        parsed = parse_observation("Alice is tall. Alice is tall.")
        props = [prop for prop, _ in observation_to_beliefs(parsed)]
        assert len(props) == len(set(props))

    def test_entity_only_observation(self):
        parsed = parse_observation("Hello Alice.")
        beliefs = observation_to_beliefs(parsed)
        assert beliefs
        assert all(conf == 0.5 for _, conf in beliefs)


class TestBelief:
    def test_defaults(self):
        belief = Belief("the sky is blue")
        assert belief.confidence == pytest.approx(0.5)
        assert belief.source == "unknown"
        assert belief.sources == ["unknown"]
        assert belief.conflict is False
        assert belief.updated >= belief.created

    def test_confidence_clamped(self):
        assert Belief("x", confidence=5.0).confidence == 1.0
        assert Belief("x", confidence=-1.0).confidence == 0.0

    def test_dict_roundtrip(self):
        belief = Belief("the sky is blue", confidence=0.8, source="perception")
        restored = Belief.from_dict(belief.to_dict())
        assert restored == belief
        assert restored.proposition == "the sky is blue"
        assert restored.sources == ["perception"]


class TestBeliefStore:
    def test_assert_and_get_new_belief(self):
        store = BeliefStore()
        belief = store.assert_belief("the sky is blue", 0.8, "perception")
        assert belief.confidence == pytest.approx(0.8)
        assert store.get("the sky is blue") is belief
        assert len(store) == 1

    def test_lookup_is_case_insensitive(self):
        store = BeliefStore()
        store.assert_belief("The Sky Is Blue", 0.8, "perception")
        assert store.get("the sky is blue") is not None
        assert "THE SKY IS BLUE" in store

    def test_reassert_blends_confidence(self):
        store = BeliefStore()
        store.assert_belief("cats are cute", 0.4, "a")
        belief = store.assert_belief("cats are cute", 0.8, "b")
        assert belief.confidence == pytest.approx(0.4 + (0.8 - 0.4) * 0.5)
        assert belief.sources == ["a", "b"]

    def test_reassert_tracks_sources_without_dupes(self):
        store = BeliefStore()
        store.assert_belief("cats are cute", 0.4, "a")
        belief = store.assert_belief("cats are cute", 0.8, "a")
        assert belief.sources == ["a"]

    def test_contradiction_lowers_both_and_marks_conflict(self):
        store = BeliefStore()
        first = store.assert_belief("the sky is blue", 0.8, "perception")
        second = store.assert_belief("the sky is green", 0.9, "rumor")
        assert first.confidence == pytest.approx(0.55)
        assert second.confidence == pytest.approx(0.65)
        assert first.conflict is True
        assert second.conflict is True

    def test_non_contradicting_asserts_leave_confidence_alone(self):
        store = BeliefStore()
        store.assert_belief("the sky is blue", 0.8, "perception")
        other = store.assert_belief("grass is green", 0.9, "perception")
        assert other.confidence == pytest.approx(0.9)
        assert other.conflict is False

    def test_contradictions_lists_pairs(self):
        store = BeliefStore()
        store.assert_belief("the sky is blue", 0.8, "a")
        store.assert_belief("the sky is green", 0.9, "b")
        store.assert_belief("grass is green", 0.7, "c")
        pairs = store.contradictions()
        assert len(pairs) == 1
        props = {pair[0].proposition for pair in pairs} | {pair[1].proposition for pair in pairs}
        assert props == {"the sky is blue", "the sky is green"}

    def test_retract(self):
        store = BeliefStore()
        store.assert_belief("the sky is blue", 0.8, "a")
        assert store.retract("the sky is blue") is True
        assert store.get("the sky is blue") is None
        assert store.retract("the sky is blue") is False

    def test_query_by_keyword(self):
        store = BeliefStore()
        store.assert_belief("roses are red", 0.9, "a")
        store.assert_belief("fire trucks are red", 0.4, "b")
        store.assert_belief("the sky is blue", 0.8, "c")
        results = store.query("red")
        assert [belief.proposition for belief in results] == [
            "roses are red",
            "fire trucks are red",
        ]
        assert store.query("purple") == []

    def test_query_min_confidence(self):
        store = BeliefStore()
        store.assert_belief("roses are red", 0.9, "a")
        store.assert_belief("fire trucks are red", 0.4, "b")
        results = store.query("red", min_confidence=0.5)
        assert [belief.proposition for belief in results] == ["roses are red"]

    def test_strongest(self):
        store = BeliefStore()
        store.assert_belief("a is b", 0.3, "x")
        store.assert_belief("c is d", 0.9, "x")
        store.assert_belief("e is f", 0.6, "x")
        top = store.strongest(2)
        assert [belief.proposition for belief in top] == ["c is d", "e is f"]
        assert len(store.strongest(100)) == 3

    def test_uses_now_callable(self):
        clock = [1000.0]
        store = BeliefStore(now=lambda: clock[0])
        belief = store.assert_belief("x is y", 0.5, "a")
        assert belief.created == pytest.approx(1000.0)
        clock[0] = 2000.0
        store.assert_belief("x is y", 0.6, "b")
        assert belief.updated == pytest.approx(2000.0)

    def test_dict_roundtrip(self):
        store = BeliefStore()
        store.assert_belief("the sky is blue", 0.8, "perception")
        store.assert_belief("the sky is green", 0.9, "rumor")
        restored = BeliefStore.from_dict(store.to_dict())
        assert len(restored) == 2
        assert restored.get("the sky is blue").confidence == pytest.approx(0.55)
        assert restored.get("the sky is green").conflict is True
        assert len(restored.contradictions()) == 1


class TestSchema:
    def test_slot_forms_normalize(self):
        schema = Schema(
            "test",
            {
                "a": "plain description",
                "b": ("tuple description", True),
                "c": {"description": "dict description", "required": True, "source": "number"},
            },
        )
        assert schema.slots["a"]["required"] is False
        assert schema.slots["a"]["source"] == "entity"
        assert schema.slots["b"]["required"] is True
        assert schema.slots["c"]["source"] == "number"

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            Schema("bad", {"a": {"description": "x", "source": "nope"}})

    def test_fill_slot_and_completion(self):
        schema = Schema("t", {"name": ("the name", True), "nick": "optional"})
        assert schema.is_complete() is False
        assert schema.missing() == ["name"]
        schema.fill_slot("name", "Alice")
        assert schema.is_complete() is True
        with pytest.raises(KeyError):
            schema.fill_slot("bogus", 1)

    def test_dict_roundtrip(self):
        schema = Schema("t", {"name": ("the name", True)})
        schema.fill_slot("name", "Alice")
        restored = Schema.from_dict(schema.to_dict())
        assert restored.name == "t"
        assert restored.fillers == {"name": "Alice"}
        assert restored.slots["name"]["required"] is True


class TestSchemaLibrary:
    def test_default_schemas_registered(self):
        library = SchemaLibrary()
        assert set(library.names()) == {"person", "object_fact", "preference", "event"}

    def test_define_and_get(self):
        library = SchemaLibrary()
        schema = library.define("custom", {"thing": "the thing"})
        assert library.get("custom") is schema
        assert "custom" in library.names()
        assert library.get("missing") is None

    def test_match_prefers_best_frame(self):
        library = SchemaLibrary()
        parsed = parse_observation("Alice is a doctor.")
        matches = library.match(parsed)
        assert matches
        assert matches[0][0].name == "object_fact"
        assert matches[0][1] == pytest.approx(1.0)
        # scores descend
        scores = [score for _, score in matches]
        assert scores == sorted(scores, reverse=True)

    def test_match_preference(self):
        library = SchemaLibrary()
        parsed = parse_observation("Bob likes pizza.")
        matches = library.match(parsed)
        assert matches[0][0].name == "preference"
        assert matches[0][1] == pytest.approx(1.0)

    def test_match_event(self):
        library = SchemaLibrary()
        parsed = parse_observation("Alice visited Paris tomorrow.")
        matches = library.match(parsed)
        assert matches[0][0].name == "event"
        assert matches[0][1] == pytest.approx(1.0)

    def test_match_empty_parsed_scores_nothing(self):
        library = SchemaLibrary()
        assert library.match(parse_observation("")) == []

    def test_fill_person(self):
        library = SchemaLibrary()
        parsed = parse_observation("Alice is a doctor.")
        filled = library.fill("person", parsed)
        assert filled["name"] == "Alice"
        assert filled["occupation"] == "doctor"
        assert filled["age"] is None

    def test_fill_preference(self):
        library = SchemaLibrary()
        parsed = parse_observation("Bob likes pizza.")
        filled = library.fill("preference", parsed)
        assert filled == {"subject": "Bob", "liked": "pizza"}

    def test_fill_event_with_dateword(self):
        library = SchemaLibrary()
        parsed = parse_observation("Alice visited Paris tomorrow.")
        filled = library.fill("event", parsed)
        assert filled == {"who": "Alice", "action": "visited", "when": "tomorrow"}

    def test_fill_unknown_schema_raises(self):
        library = SchemaLibrary()
        with pytest.raises(KeyError):
            library.fill("nope", parse_observation("hello"))

    def test_custom_schema_matches_and_fills(self):
        library = SchemaLibrary()
        library.define(
            "purchase",
            {
                "buyer": {"description": "who bought", "required": True, "source": "entity"},
                "price": {"description": "what it cost", "required": True, "source": "number"},
            },
        )
        parsed = parse_observation("Alice bought a bike for 250 dollars.")
        matches = library.match(parsed)
        assert matches[0][0].name == "purchase"
        filled = library.fill("purchase", parsed)
        assert filled["buyer"] == "Alice"
        assert filled["price"] == 250

    def test_dict_roundtrip_keeps_defaults_and_custom(self):
        library = SchemaLibrary()
        library.define("custom", {"thing": "the thing"})
        restored = SchemaLibrary.from_dict(library.to_dict())
        assert set(restored.names()) == set(library.names())
        assert restored.get("person").slots["name"]["required"] is True
        parsed = parse_observation("Alice is a doctor.")
        assert restored.fill("person", parsed)["name"] == "Alice"


class TestPipelineEdgeCases:
    def test_numbers_with_commas_and_signs(self):
        parsed = parse_observation("the debt grew by 1,000 dollars and fell -5 points")
        values = [item["value"] for item in parsed["numbers"]]
        assert 1000 in values
        assert -5 in values

    def test_pronoun_subject_relations(self):
        parsed = parse_observation("I like pizza.")
        assert ("I", "like", "pizza") in parsed["relations"]

    def test_multiple_relations(self):
        parsed = parse_observation("Alice likes pizza. Bob has a cat.")
        assert ("Alice", "likes", "pizza") in parsed["relations"]
        assert ("Bob", "has", "cat") in parsed["relations"]

    def test_intent_question_word_without_mark(self):
        assert parse_observation("What time is it")["intent"] == "question"

    def test_observation_to_beliefs_lowercases(self):
        parsed = parse_observation("NASA launched Artemis.")
        props = [prop for prop, _ in observation_to_beliefs(parsed)]
        assert props
        assert all(prop == prop.lower() for prop in props)


class TestBeliefStoreMerge:
    def test_merge_combines_stores(self):
        first = BeliefStore()
        first.assert_belief("the sky is blue", 0.8, "a")
        second = BeliefStore()
        second.assert_belief("grass is green", 0.9, "b")
        second.assert_belief("the sky is blue", 0.6, "c")
        first.merge(second)
        assert len(first) == 2
        merged = first.get("the sky is blue")
        assert set(merged.sources) == {"a", "c"}
        assert merged.confidence == pytest.approx(0.8 + (0.6 - 0.8) * 0.5)

    def test_merge_preserves_conflict(self):
        first = BeliefStore()
        second = BeliefStore()
        second.assert_belief("the sky is blue", 0.8, "a")
        second.assert_belief("the sky is green", 0.9, "b")
        first.merge(second)
        assert first.get("the sky is blue").conflict is True
        assert len(first.contradictions()) == 1

    def test_clear(self):
        store = BeliefStore()
        store.assert_belief("x is y", 0.5, "a")
        store.clear()
        assert len(store) == 0
        assert store.get("x is y") is None

    def test_query_is_case_insensitive(self):
        store = BeliefStore()
        store.assert_belief("the sky is blue", 0.8, "a")
        assert store.query("SKY") != []


class TestSchemaExtras:
    def test_describe(self):
        schema = Schema("t", {"name": ("the name", True), "nick": "optional"})
        text = schema.describe()
        assert "schema t:" in text
        assert "name (required)" in text
        assert "nick (optional)" in text

    def test_unregister(self):
        library = SchemaLibrary()
        assert library.unregister("person") is True
        assert library.get("person") is None
        assert library.unregister("person") is False

    def test_fill_all(self):
        library = SchemaLibrary()
        parsed = parse_observation("Bob likes pizza.")
        filled = library.fill_all(parsed)
        assert filled["preference"] == {"subject": "Bob", "liked": "pizza"}
        assert set(filled) == {schema.name for schema, _ in library.match(parsed)}

    def test_keyword_text_and_indexed_entity_sources(self):
        library = SchemaLibrary()
        library.define(
            "note",
            {
                "topic": {"description": "main topic", "source": "keyword"},
                "full": {"description": "whole text", "source": "text"},
                "second_entity": {
                    "description": "runner up",
                    "source": "entity",
                    "index": 1,
                },
            },
        )
        parsed = parse_observation("Alice met Bob in Paris.")
        filled = library.fill("note", parsed)
        assert filled["topic"] == parsed["keywords"][0]
        assert filled["full"] == "Alice met Bob in Paris."
        assert filled["second_entity"] == "Bob"
