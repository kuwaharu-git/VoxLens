"""
Speaker diarization module using pyannote.audio
"""
from typing import List, Tuple
import os
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
                             If not provided, will try to read from HF_TOKEN environment variable
        """
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        # Use provided token, fallback to environment variable
        self.huggingface_token = huggingface_token or os.getenv("HF_TOKEN")
        
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
        
        # Clear VRAM cache after inference to optimize memory usage
        self.clear_cache()
        
        return segments
    
    def clear_cache(self):
        """Clear GPU cache to free VRAM after inference"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def cleanup(self):
        """Cleanup method to release resources and clear VRAM"""
        if self.pipeline is not None:
            # Move pipeline to CPU before deletion to free GPU memory
            if hasattr(self.pipeline, 'to'):
                self.pipeline.to(torch.device('cpu'))
            del self.pipeline
            self.pipeline = None
        self.clear_cache()
