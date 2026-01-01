"""Unit tests for TTL parsing and injection."""

from ttl_processor import TTLProcessor


class DummySettings:  # pylint: disable=too-few-public-methods
    """Simple settings container for TTL tests."""

    # pylint: disable=too-few-public-methods
    default_ttl_seconds = 5
    unload_ttl_seconds = 2


def test_ttl_parse_numeric() -> None:
    """Parse numeric keep_alive values."""
    processor = TTLProcessor(DummySettings())
    assert processor.parse_keep_alive_to_ttl_seconds(10) == 10


def test_ttl_parse_duration() -> None:
    """Parse duration strings into seconds."""
    processor = TTLProcessor(DummySettings())
    assert processor.parse_keep_alive_to_ttl_seconds("5m") == 300


def test_ttl_parse_unload() -> None:
    """Parse unload values into unload TTL."""
    processor = TTLProcessor(DummySettings())
    assert processor.parse_keep_alive_to_ttl_seconds(0) == 2


def test_ttl_inject_default() -> None:
    """Inject default TTL when keep_alive is not provided."""
    processor = TTLProcessor(DummySettings())
    payload = {}
    updated = processor.inject_ttl(payload, keep_alive=None)
    assert updated["ttl"] == 5
