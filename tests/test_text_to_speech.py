import pytest
from unittest.mock import Mock, patch, mock_open
import os
from pathlib import Path
from pydub import AudioSegment
from google.cloud import texttospeech

from podcast_llm.text_to_speech import (
    clean_text_for_tts,
    merge_audio_files,
    process_line_google,
    generate_audio,
    rate_limit_per_minute
)

# Test data
SAMPLE_LINES = [
    {'speaker': 'Interviewer', 'text': 'Hello *world* with _emphasis_ and—dash'},
    {'speaker': 'Interviewee', 'text': 'Hi there *friend*!'}
]

CLEANED_LINES = [
    {'speaker': 'Interviewer', 'text': 'Hello world with emphasis anddash'},
    {'speaker': 'Interviewee', 'text': 'Hi there friend!'}
]

@pytest.fixture
def mock_audio_segment():
    with patch('podcast_llm.text_to_speech.AudioSegment') as mock:
        mock.empty.return_value = Mock()
        mock.from_file.return_value = Mock()
        yield mock

@pytest.fixture
def mock_tts_client():
    with patch('podcast_llm.text_to_speech.texttospeech.TextToSpeechClient') as mock:
        mock_instance = Mock()
        mock_instance.synthesize_speech.return_value = Mock(audio_content=b'fake_audio')
        mock.return_value = mock_instance
        yield mock_instance


def test_clean_text_for_tts():
    """Test that special characters are properly removed from text"""
    result = clean_text_for_tts(SAMPLE_LINES)
    assert result == CLEANED_LINES

