"""
Transcription module using faster-whisper
"""
from typing import List, Tuple
from faster_whisper import WhisperModel
import config


class AudioTranscriber:
    """Audio transcription using faster-whisper"""
    
    def __init__(self):
        """Initialize the transcription model"""
        self.model = None
        
    def load_model(self):
        """Load the faster-whisper model"""
        if self.model is None:
            device = config.DEVICE if config.DEVICE == "cuda" else "cpu"
            self.model = WhisperModel(
                config.TRANSCRIPTION_MODEL,
                device=device,
                compute_type=config.COMPUTE_TYPE if device == "cuda" else "int8"
            )
    
    def transcribe_segment(self, audio_path: str, start_time: float, end_time: float) -> str:
        """
        Transcribe a specific segment of audio
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Transcribed text for the segment
        """
        if self.model is None:
            self.load_model()
        
        # Transcribe with timestamps
        segments, _ = self.model.transcribe(
            audio_path,
            language="ja",  # Japanese language
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Filter segments within the time range
        transcription_parts = []
        for segment in segments:
            # Check if segment overlaps with our time range
            if segment.start >= start_time and segment.end <= end_time:
                transcription_parts.append(segment.text.strip())
            elif segment.start < end_time and segment.end > start_time:
                # Partial overlap
                transcription_parts.append(segment.text.strip())
        
        return " ".join(transcription_parts)
    
    def transcribe_with_speakers(
        self, 
        audio_path: str, 
        speaker_segments: List[Tuple[float, float, str]]
    ) -> str:
        """
        Transcribe audio with speaker labels
        
        Args:
            audio_path: Path to audio file
            speaker_segments: List of (start_time, end_time, speaker_label) tuples
            
        Returns:
            Full transcription with speaker labels in "SPEAKER_XX: text" format
        """
        if self.model is None:
            self.load_model()
        
        full_transcription = []
        
        for start_time, end_time, speaker_label in speaker_segments:
            # Transcribe this segment
            text = self.transcribe_segment(audio_path, start_time, end_time)
            
            if text.strip():  # Only add non-empty transcriptions
                # Format as "SPEAKER_XX: text"
                full_transcription.append(f"{speaker_label}: {text}")
        
        return "\n".join(full_transcription)
