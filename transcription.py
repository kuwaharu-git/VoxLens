"""
Transcription module using faster-whisper
"""
from typing import List, Tuple
import torch
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
        
        # Transcribe the entire audio once
        segments, _ = self.model.transcribe(
            audio_path,
            language=config.TRANSCRIPTION_LANGUAGE,
            beam_size=config.BEAM_SIZE,
            vad_filter=config.VAD_FILTER,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Convert segments to a list for easier processing
        all_segments = list(segments)
        
        full_transcription = []
        
        # Match transcribed segments with speaker segments
        for start_time, end_time, speaker_label in speaker_segments:
            # Find all transcription segments that overlap with this speaker segment
            segment_texts = []
            
            for segment in all_segments:
                # Check if segment overlaps with speaker time range
                if segment.start < end_time and segment.end > start_time:
                    segment_texts.append(segment.text.strip())
            
            if segment_texts:  # Only add non-empty transcriptions
                # Format as "SPEAKER_XX: text"
                combined_text = " ".join(segment_texts)
                full_transcription.append(f"{speaker_label}: {combined_text}")
        
        # Clear VRAM cache after inference to optimize memory usage
        self.clear_cache()
        
        return "\n".join(full_transcription)
    
    def clear_cache(self):
        """Clear GPU cache to free VRAM after inference"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def cleanup(self):
        """Cleanup method to release resources and clear VRAM"""
        if self.model is not None:
            # faster-whisper uses CTranslate2 backend which doesn't have .to() method
            # Simply delete the model object to free resources
            del self.model
            self.model = None
        self.clear_cache()
