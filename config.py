"""
Configuration file for VoxLens application
"""

# Model configurations
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Transcription model size (faster-whisper)
# Available options: tiny, base, small, medium, large, large-v2, large-v3, distil-large-v2, distil-large-v3
TRANSCRIPTION_MODEL = "distil-large-v3"

# Summarization model size (Ollama)
# Available options: llama3.2:1b, llama3.2:3b, llama3.2:8b, llama3.1:8b, llama3.1:70b
# Make sure the model is pulled with 'ollama pull <model_name>'
LLM_MODEL = "llama3.2:8b"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"

# Processing settings
DEVICE = "cuda"  # or "cpu" if CUDA is not available
COMPUTE_TYPE = "float16"  # for faster-whisper

# Transcription settings
TRANSCRIPTION_LANGUAGE = "ja"  # Language code (ja, en, zh, etc.)
BEAM_SIZE = 5
VAD_FILTER = True

# Summarization settings
MAX_STUFF_CHAIN_LENGTH = 4000  # Maximum character length for StuffDocumentsChain

# Audio settings
SUPPORTED_FORMATS = ["mp3", "wav"]
