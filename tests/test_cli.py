from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sermon_finder.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def _make_cm(return_value):
    """Build a mock context manager that yields return_value."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=return_value)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _mock_pipeline(t: float = 180.0):
    """Return patch targets for the diarize-first happy path.

    t is the diarization transition timestamp in seconds.
    Expected output: (int(t) // 60)'(int(t) % 60):02d
    """
    return {
        "sermon_finder.cli.audio.prepare_audio": MagicMock(return_value=_make_cm("/tmp/audio.wav")),
        "sermon_finder.cli.audio.split_wav": MagicMock(return_value=_make_cm([("/tmp/c0.wav", 0.0, None)])),
        "sermon_finder.cli._diarizer.run_diarization": MagicMock(
            return_value=[
                {"speaker": "A", "start": 0.0, "end": t - 1},
                {"speaker": "B", "start": t, "end": t + 10},
            ]
        ),
        "sermon_finder.cli._diarizer.get_speaker_transitions": MagicMock(return_value=[t]),
        "sermon_finder.cli.audio.extract_window": MagicMock(return_value=_make_cm(("/tmp/win.wav", max(0.0, t - 30.0)))),
        "sermon_finder.cli.transcriber.transcribe_segment": MagicMock(
            return_value=[{"start": t, "end": t + 5, "text": "test"}]
        ),
        "sermon_finder.cli.analyzer.is_sermon_transition": MagicMock(return_value=True),
    }


def _run_with_mocks(runner, args, mocks, env=None):
    with ExitStack() as stack:
        for target, mock in mocks.items():
            stack.enter_context(patch(target, mock))
        return runner.invoke(main, args, env=env or {"ANTHROPIC_API_KEY": "test"})


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
    result = _run_with_mocks(runner, [sample_wav], _mock_pipeline(t=180.0))
    assert result.exit_code == 0
    assert "3'00" in result.output


def test_timestamp_format_zero_padded_seconds(runner, sample_wav):
    result = _run_with_mocks(runner, [sample_wav], _mock_pipeline(t=725.0))
    assert "12'05" in result.output


def test_sermon_not_found_exits_with_error(runner, sample_wav):
    mocks = {
        "sermon_finder.cli.audio.prepare_audio": MagicMock(return_value=_make_cm("/tmp/audio.wav")),
        "sermon_finder.cli.audio.split_wav": MagicMock(return_value=_make_cm([("/tmp/c0.wav", 0.0, None)])),
        "sermon_finder.cli._diarizer.run_diarization": MagicMock(return_value=[]),
        "sermon_finder.cli._diarizer.get_speaker_transitions": MagicMock(return_value=[]),
    }
    result = _run_with_mocks(runner, [sample_wav], mocks)
    assert result.exit_code == 1
    assert "Could not find" in result.output
