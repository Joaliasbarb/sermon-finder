from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sermon_finder.analyzer import TransitionResult
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


def _ok() -> TransitionResult:
    return TransitionResult(is_sermon=True, uncertain=False, quality_ok=True)


def _rejected() -> TransitionResult:
    return TransitionResult(is_sermon=False, uncertain=False, quality_ok=True)


def _poor() -> TransitionResult:
    return TransitionResult(is_sermon=False, uncertain=False, quality_ok=False)


def _mock_pipeline(t: float = 180.0):
    """Return patch targets for the diarize-first happy path.

    t is the diarization transition timestamp in seconds.
    Expected output: (int(t) // 60)'(int(t) % 60):02d
    """
    return {
        "sermon_finder.cli.audio.prepare_audio": MagicMock(return_value=_make_cm("/tmp/audio.wav")),
        "sermon_finder.cli.audio.split_wav": MagicMock(return_value=_make_cm([("/tmp/c0.wav", 0.0, None)])),
        "sermon_finder.diarizer.run_diarization": MagicMock(
            return_value=[
                {"speaker": "A", "start": 0.0, "end": t - 1},
                {"speaker": "B", "start": t, "end": t + 10},
            ]
        ),
        "sermon_finder.diarizer.get_speaker_transitions": MagicMock(return_value=[t]),
        "sermon_finder.audio.extract_window": MagicMock(return_value=_make_cm(("/tmp/win.wav", max(0.0, t - 30.0)))),
        "sermon_finder.transcriber.transcribe_segment": MagicMock(
            return_value=[{"start": t, "end": t + 5, "text": "test"}]
        ),
        "sermon_finder.analyzer.is_sermon_transition": MagicMock(return_value=_ok()),
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
        "sermon_finder.diarizer.run_diarization": MagicMock(return_value=[]),
        "sermon_finder.diarizer.get_speaker_transitions": MagicMock(return_value=[]),
    }
    result = _run_with_mocks(runner, [sample_wav], mocks)
    assert result.exit_code == 1
    assert "Could not find" in result.output


def _base_mocks(t: float, transcribe_mock, llm_mock):
    return {
        "sermon_finder.cli.audio.prepare_audio": MagicMock(return_value=_make_cm("/tmp/audio.wav")),
        "sermon_finder.cli.audio.split_wav": MagicMock(return_value=_make_cm([("/tmp/c0.wav", 0.0, None)])),
        "sermon_finder.diarizer.run_diarization": MagicMock(
            return_value=[
                {"speaker": "A", "start": 0.0, "end": t - 1},
                {"speaker": "B", "start": t, "end": t + 10},
            ]
        ),
        "sermon_finder.diarizer.get_speaker_transitions": MagicMock(return_value=[t]),
        "sermon_finder.audio.extract_window": MagicMock(return_value=_make_cm(("/tmp/win.wav", t - 30.0))),
        "sermon_finder.transcriber.transcribe_segment": transcribe_mock,
        "sermon_finder.analyzer.is_sermon_transition": llm_mock,
    }


def test_poor_quality_no_retry_without_retry_model(runner, sample_wav):
    """Without --retry-model, poor quality never triggers a retry."""
    t = 180.0
    transcribe_mock = MagicMock(return_value=[{"start": t, "end": t + 5, "text": "test"}])
    llm_mock = MagicMock(return_value=_poor())

    result = _run_with_mocks(runner, [sample_wav, "--model", "small"], _base_mocks(t, transcribe_mock, llm_mock))

    assert result.exit_code == 1
    assert transcribe_mock.call_count == 1


def test_poor_quality_triggers_retry_with_next_model(runner, sample_wav):
    """When LLM reports POOR+NO and --retry-model allows it, retry with the next model."""
    t = 180.0
    transcribe_mock = MagicMock(return_value=[{"start": t, "end": t + 5, "text": "test"}])
    llm_mock = MagicMock(side_effect=[_poor(), _ok()])

    result = _run_with_mocks(
        runner,
        [sample_wav, "--model", "small", "--retry-model", "medium"],
        _base_mocks(t, transcribe_mock, llm_mock),
    )

    assert result.exit_code == 0
    assert "3'00" in result.output
    assert transcribe_mock.call_count == 2
    assert transcribe_mock.call_args_list[1].kwargs["model_size"] == "medium"


def test_poor_quality_retries_through_multiple_models(runner, sample_wav):
    """With a wide --retry-model cap, retries step up one model at a time."""
    t = 180.0
    transcribe_mock = MagicMock(return_value=[{"start": t, "end": t + 5, "text": "test"}])
    # small → POOR, medium → POOR, large-v3 → OK
    llm_mock = MagicMock(side_effect=[_poor(), _poor(), _ok()])

    result = _run_with_mocks(
        runner,
        [sample_wav, "--model", "small", "--retry-model", "large-v3"],
        _base_mocks(t, transcribe_mock, llm_mock),
    )

    assert result.exit_code == 0
    assert "3'00" in result.output
    assert transcribe_mock.call_count == 3
    used_models = [call.kwargs["model_size"] for call in transcribe_mock.call_args_list]
    assert used_models == ["small", "medium", "large-v3"]
