"""
Speaker diarization module using pyannote.audio
"""
from typing import List, Tuple
import torch
from pyannote.audio import Pipeline
import config


class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio"""
    
    def __init__(self, huggingface_token: str = None):
        """
        Initialize the speaker diarization pipeline
        
        Args:
            huggingface_token: HuggingFace access token for model download
        """
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.huggingface_token = huggingface_token
        
    def load_model(self):
        """Load the diarization model"""
        if self.pipeline is None:
            self.pipeline = Pipeline.from_pretrained(
                config.DIARIZATION_MODEL,
                use_auth_token=self.huggingface_token
            )
            self.pipeline.to(self.device)
    
    def diarize(self, audio_path: str) -> List[Tuple[float, float, str]]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of tuples (start_time, end_time, speaker_label)
        """
        if self.pipeline is None:
            self.load_model()
        
        # Run diarization
        diarization = self.pipeline(audio_path)
        
        # Extract segments with speaker labels
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        
        return segments
