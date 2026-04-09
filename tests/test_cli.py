from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sermon_finder.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def _mock_pipeline(minutes: int = 35, seconds: int = 42):
    """Context managers + return values for the full happy path."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value="/tmp/audio.wav")
    cm.__exit__ = MagicMock(return_value=False)
    return {
        "sermon_finder.audio.prepare_audio": MagicMock(return_value=cm),
        "sermon_finder.transcriber.transcribe": MagicMock(
            return_value=[{"start": 0, "end": 5, "text": "test"}]
        ),
        "sermon_finder.analyzer.find_sermon_start": MagicMock(
            return_value=(minutes, seconds)
        ),
    }


def test_missing_api_key(runner, sample_wav, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(main, [sample_wav])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_missing_file(runner):
    result = runner.invoke(main, ["/nonexistent/audio.mp3"], env={"ANTHROPIC_API_KEY": "test"})
    assert result.exit_code == 1
    assert "File not found" in result.output


def test_happy_path_stdout(runner, sample_wav):
    mocks = _mock_pipeline(35, 42)
    with patch("sermon_finder.cli.audio.prepare_audio", mocks["sermon_finder.audio.prepare_audio"]), \
         patch("sermon_finder.cli.transcriber.transcribe", mocks["sermon_finder.transcriber.transcribe"]), \
         patch("sermon_finder.cli.analyzer.find_sermon_start", mocks["sermon_finder.analyzer.find_sermon_start"]):
        result = runner.invoke(main, [sample_wav], env={"ANTHROPIC_API_KEY": "test"})
    assert result.exit_code == 0
    assert "35'42" in result.output


def test_timestamp_format_zero_padded_seconds(runner, sample_wav):
    mocks = _mock_pipeline(12, 5)
    with patch("sermon_finder.cli.audio.prepare_audio", mocks["sermon_finder.audio.prepare_audio"]), \
         patch("sermon_finder.cli.transcriber.transcribe", mocks["sermon_finder.transcriber.transcribe"]), \
         patch("sermon_finder.cli.analyzer.find_sermon_start", mocks["sermon_finder.analyzer.find_sermon_start"]):
        result = runner.invoke(main, [sample_wav], env={"ANTHROPIC_API_KEY": "test"})
    assert "12'05" in result.output


def test_sermon_not_found_exits_with_error(runner, sample_wav):
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value="/tmp/audio.wav")
    cm.__exit__ = MagicMock(return_value=False)
    with patch("sermon_finder.cli.audio.prepare_audio", return_value=cm), \
         patch("sermon_finder.cli.transcriber.transcribe", return_value=[]), \
         patch("sermon_finder.cli.analyzer.find_sermon_start", side_effect=ValueError("Could not find")):
        result = runner.invoke(main, [sample_wav], env={"ANTHROPIC_API_KEY": "test"})
    assert result.exit_code == 1
    assert "Could not find" in result.output
