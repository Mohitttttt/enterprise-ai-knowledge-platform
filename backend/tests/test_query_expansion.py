import pytest

from app.services.query_expansion import expand_tokens


def test_expand_laptop_synonyms():
    result = expand_tokens(["laptop"])

    assert result == [
        "laptop",
        "laptops",
        "device",
        "devices",
    ]


def test_expand_preserves_original_tokens():
    original = ["company", "laptop", "stops", "working"]

    result = expand_tokens(original)

    assert result[:len(original)] == original


def test_expand_preserves_repeated_original_tokens():
    result = expand_tokens(["laptop", "laptop"])

    assert result[:2] == ["laptop", "laptop"]
    assert result.count("laptops") == 1


def test_expand_does_not_duplicate_existing_synonyms():
    result = expand_tokens(["laptop", "device"])

    assert result.count("device") == 1
    assert result.count("devices") == 1


def test_expand_multiple_keywords_without_duplicate_synonyms():
    result = expand_tokens(["stops", "breaks"])

    assert result == [
        "stops",
        "breaks",
        "problem",
        "problems",
        "malfunction",
        "repair",
        "replacement",
    ]


def test_expand_query_without_matching_synonyms():
    original = ["employee", "benefits", "policy"]

    result = expand_tokens(original)

    assert result == original


def test_expand_empty_tokens():
    assert expand_tokens([]) == []
