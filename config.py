"""
Configuration file for VoxLens application
"""

# Model configurations
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
TRANSCRIPTION_MODEL = "distil-large-v3"
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
