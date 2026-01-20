"""
Simple unit tests for VoxLens modules
Run with: python -m pytest test_modules.py
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


class TestSpeakerDiarizer:
    """Tests for SpeakerDiarizer class"""
    
    def test_init(self):
        """Test SpeakerDiarizer initialization"""
        from diarization import SpeakerDiarizer
        
        diarizer = SpeakerDiarizer(huggingface_token="test_token")
        assert diarizer.huggingface_token == "test_token"
        assert diarizer.pipeline is None
    
    def test_init_with_env_variable(self):
        """Test SpeakerDiarizer initialization with HF_TOKEN environment variable"""
        from diarization import SpeakerDiarizer
        
        with patch.dict(os.environ, {'HF_TOKEN': 'env_token'}):
            diarizer = SpeakerDiarizer()
            assert diarizer.huggingface_token == "env_token"
        
        # Provided token should override environment variable
        with patch.dict(os.environ, {'HF_TOKEN': 'env_token'}):
            diarizer = SpeakerDiarizer(huggingface_token="provided_token")
            assert diarizer.huggingface_token == "provided_token"
    
    @patch('diarization.Pipeline')
    def test_load_model(self, mock_pipeline):
        """Test model loading"""
        from diarization import SpeakerDiarizer
        
        diarizer = SpeakerDiarizer(huggingface_token="test_token")
        diarizer.load_model()
        
        mock_pipeline.from_pretrained.assert_called_once()
        assert diarizer.pipeline is not None
    
    @patch('diarization.torch')
    def test_clear_cache(self, mock_torch):
        """Test VRAM cache clearing"""
        from diarization import SpeakerDiarizer
        
        mock_torch.cuda.is_available.return_value = True
        diarizer = SpeakerDiarizer(huggingface_token="test_token")
        diarizer.clear_cache()
        
        mock_torch.cuda.empty_cache.assert_called_once()
    
    @patch('diarization.torch')
    def test_cleanup(self, mock_torch):
        """Test cleanup method"""
        from diarization import SpeakerDiarizer
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        
        diarizer = SpeakerDiarizer(huggingface_token="test_token")
        mock_pipeline = MagicMock()
        diarizer.pipeline = mock_pipeline
        
        diarizer.cleanup()
        
        assert diarizer.pipeline is None
        mock_torch.cuda.empty_cache.assert_called()


class TestAudioTranscriber:
    """Tests for AudioTranscriber class"""
    
    def test_init(self):
        """Test AudioTranscriber initialization"""
        from transcription import AudioTranscriber
        
        transcriber = AudioTranscriber()
        assert transcriber.model is None
    
    @patch('transcription.WhisperModel')
    def test_load_model(self, mock_whisper):
        """Test model loading"""
        from transcription import AudioTranscriber
        
        transcriber = AudioTranscriber()
        transcriber.load_model()
        
        mock_whisper.assert_called_once()
        assert transcriber.model is not None
    
    @patch('transcription.torch')
    def test_clear_cache(self, mock_torch):
        """Test VRAM cache clearing"""
        from transcription import AudioTranscriber
        
        mock_torch.cuda.is_available.return_value = True
        transcriber = AudioTranscriber()
        transcriber.clear_cache()
        
        mock_torch.cuda.empty_cache.assert_called_once()
    
    @patch('transcription.torch')
    def test_cleanup(self, mock_torch):
        """Test cleanup method"""
        from transcription import AudioTranscriber
        
        mock_torch.cuda.is_available.return_value = True
        
        transcriber = AudioTranscriber()
        transcriber.model = MagicMock()
        
        transcriber.cleanup()
        
        assert transcriber.model is None
        mock_torch.cuda.empty_cache.assert_called()


class TestConversationSummarizer:
    """Tests for ConversationSummarizer class"""
    
    def test_init(self):
        """Test ConversationSummarizer initialization"""
        from summarization import ConversationSummarizer
        
        summarizer = ConversationSummarizer()
        assert summarizer.llm is None
    
    @patch('summarization.Ollama')
    def test_initialize_llm(self, mock_ollama):
        """Test LLM initialization"""
        from summarization import ConversationSummarizer
        
        summarizer = ConversationSummarizer()
        summarizer._initialize_llm()
        
        mock_ollama.assert_called_once()
        assert summarizer.llm is not None
    
    @patch('summarization.Ollama')
    @patch('summarization.StuffDocumentsChain')
    def test_summarize(self, mock_chain, mock_ollama):
        """Test summarization"""
        from summarization import ConversationSummarizer
        
        # Mock the chain run method
        mock_chain_instance = MagicMock()
        mock_chain_instance.run.return_value = "Test summary"
        mock_chain.return_value = mock_chain_instance
        
        summarizer = ConversationSummarizer()
        
        # Mock the LLM
        summarizer.llm = MagicMock()
        
        # Test transcription
        transcription = "SPEAKER_00: Hello\nSPEAKER_01: Hi there"
        result = summarizer.summarize(transcription, use_map_reduce=False)
        
        assert isinstance(result, str)


class TestConfig:
    """Tests for configuration"""
    
    def test_config_values(self):
        """Test that config values are set"""
        import config
        
        assert hasattr(config, 'DIARIZATION_MODEL')
        assert hasattr(config, 'TRANSCRIPTION_MODEL')
        assert hasattr(config, 'LLM_MODEL')
        assert hasattr(config, 'SUPPORTED_FORMATS')
        
        assert config.DIARIZATION_MODEL == "pyannote/speaker-diarization-3.1"
        assert config.TRANSCRIPTION_MODEL == "distil-large-v3"
        assert config.LLM_MODEL == "llama3.2:8b"
        assert "mp3" in config.SUPPORTED_FORMATS
        assert "wav" in config.SUPPORTED_FORMATS
    
    def test_compute_type_float16(self):
        """Test that compute_type is set to float16 for VRAM optimization"""
        import config
        
        assert hasattr(config, 'COMPUTE_TYPE')
        assert config.COMPUTE_TYPE == "float16"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
