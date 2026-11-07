import pytest
from pydantic import ValidationError

from app.protocol import AudioChunk, SessionControl, VideoFrame, parse_client_message


def test_parses_each_client_message_type():
    assert isinstance(parse_client_message('{"type":"video_frame","ts":1,"jpeg_b64":"AA=="}'), VideoFrame)
    chunk = parse_client_message('{"type":"audio_chunk","ts":1,"pcm16_b64":"AAA="}')
    assert isinstance(chunk, AudioChunk) and chunk.sample_rate == 16000
    control = parse_client_message('{"type":"session","action":"start","mode":"hybrid"}')
    assert isinstance(control, SessionControl) and control.mode == "hybrid"


@pytest.mark.parametrize("raw", [
    '{"type":"unknown","ts":1}',
    '{"type":"session","action":"pause"}',
    '{"type":"session","action":"start","mode":"gpt"}',
    '{"type":"video_frame","ts":1}',
    'not json',
])
def test_rejects_invalid_messages(raw):
    with pytest.raises(ValidationError):
        parse_client_message(raw)
