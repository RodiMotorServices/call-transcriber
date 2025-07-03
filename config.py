"""
Configuration settings for Call Transcriber
Customize these settings based on your call center needs
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Speaker Diarization Settings
SPEAKER_SETTINGS = {
    # Minimum pause duration (seconds) to trigger speaker change
    "pause_threshold": 2.0,
    
    # Assume agent speaks first (True/False)
    "agent_speaks_first": True,
    
    # Speaker labels
    "agent_label": "AGENT",
    "client_label": "CLIENT",
    
    # Alternative labels for different use cases
    "alternative_labels": {
        "support": {"agent": "SUPPORT_AGENT", "client": "CUSTOMER"},
        "sales": {"agent": "SALES_REP", "client": "PROSPECT"},
        "technical": {"agent": "TECHNICIAN", "client": "USER"},
        "spanish": {"agent": "AGENTE", "client": "CLIENTE"},
        "support_es": {"agent": "AGENTE_SOPORTE", "client": "CLIENTE"}
    }
}

# Audio Processing Settings
AUDIO_SETTINGS = {
    # Target sample rate for processing (Hz)
    "sample_rate": 16000,
    
    # Force mono audio (recommended for speech)
    "force_mono": True,
    
    # Audio normalization
    "normalize_audio": True,
    
    # Noise reduction (basic)
    "reduce_noise": False,
    
    # Audio quality improvements for call center audio
    "telephony_optimization": True,
    "volume_normalization": True,
    "silence_removal": False
}

# Whisper Model Settings
WHISPER_SETTINGS = {
    # Default model size
    "default_model": "base",
    
    # Available models with their characteristics
    "models": {
        "tiny": {"size_mb": 39, "speed": "fastest", "accuracy": "good", "languages": "limited"},
        "base": {"size_mb": 74, "speed": "fast", "accuracy": "better", "languages": "good"},
        "small": {"size_mb": 244, "speed": "medium", "accuracy": "good", "languages": "very_good"},
        "medium": {"size_mb": 769, "speed": "slow", "accuracy": "very_good", "languages": "excellent"},
        "large": {"size_mb": 1550, "speed": "slowest", "accuracy": "excellent", "languages": "excellent"}
    },
    
    # Default language (None for auto-detection)
    "default_language": "auto",
    
    # Temperature for transcription (0.0 = deterministic, 1.0 = creative)
    "temperature": 0.0,
    
    # Beam size for decoding
    "beam_size": 5,
    
    # Best of parameter
    "best_of": 5,
    
    # Language-specific optimizations
    "language_optimizations": {
        "es": {
            "recommended_model": "medium",
            "temperature": 0.0,
            "beam_size": 5,
            "pause_threshold": 1.8
        },
        "en": {
            "recommended_model": "base",
            "temperature": 0.0,
            "beam_size": 5,
            "pause_threshold": 2.0
        }
    }
}

# Output Format Settings
OUTPUT_SETTINGS = {
    # Include timestamps in output
    "include_timestamps": True,
    
    # Include confidence scores (if available)
    "include_confidence": True,
    
    # Include word-level timestamps
    "word_timestamps": False,
    
    # JSON formatting
    "json_indent": 2,
    "ensure_ascii": False,
    
    # Additional metadata to include
    "include_model_info": True,
    "include_processing_time": True,
    "include_file_info": True,
    "include_language_detection": True
}

# Google Gemini API Settings
GEMINI_SETTINGS = {
    # API Key (loaded from environment)
    "api_key": os.getenv("GEMINI_API_KEY"),
    
    # Default model to use
    "default_model": "gemini-1.5-flash",
    
    # Available models with their characteristics
    "models": {
        "gemini-1.5-flash": {
            "description": "Fast and efficient model for most tasks",
            "max_tokens": 1048576,
            "supports_multimodal": True,
            "rate_limit": "15 RPM",
            "cost": "low"
        },
        "gemini-1.5-pro": {
            "description": "More capable model for complex tasks",
            "max_tokens": 2097152,
            "supports_multimodal": True,
            "rate_limit": "2 RPM",
            "cost": "higher"
        },
        "gemini-1.0-pro": {
            "description": "Previous generation model",
            "max_tokens": 32768,
            "supports_multimodal": False,
            "rate_limit": "60 RPM",
            "cost": "low"
        }
    },
    
    # Generation parameters
    "generation_config": {
        "temperature": 0.1,  # Low temperature for consistent results
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192,
        "candidate_count": 1
    },
    
    # Safety settings (use None for minimal blocking)
    "safety_settings": None,
    
    # Request settings
    "timeout": 60,  # seconds
    "retry_attempts": 3,
    "retry_delay": 1.0,  # seconds
    
    # Text processing
    "max_chunk_size": 30000,  # characters
    "chunk_overlap": 500,     # characters
    "preserve_context": True
}

# AI Integration Settings
AI_SETTINGS = {
    # Maximum text length for AI processing chunks
    "max_chunk_length": 4000,
    
    # Include conversation context
    "include_context": True,
    
    # Sentiment analysis placeholder (for future integration)
    "enable_sentiment": False,
    
    # Keywords extraction (for future integration)
    "extract_keywords": False,
    
    # Call classification (for future integration)
    "classify_call_type": False,
    
    # Language-aware processing
    "language_aware_chunking": True,
    "preserve_language_context": True,
    
    # Gemini-specific settings
    "use_gemini": True,
    "gemini_model": GEMINI_SETTINGS["default_model"]
}

# Call Center Specific Settings
CALL_CENTER_SETTINGS = {
    # Expected call types
    "call_types": [
        "support", "sales", "technical", "billing", "complaint", "inquiry"
    ],
    
    # Quality metrics to calculate
    "quality_metrics": {
        "talk_time_ratio": True,
        "silence_detection": True,
        "interruption_count": True,
        "response_time": True,
        "language_consistency": True
    },
    
    # Compliance features
    "compliance": {
        "redact_personal_info": False,  # Future feature
        "mark_sensitive_topics": False,  # Future feature
        "call_recording_notice": True
    },
    
    # Multi-language support
    "multilingual": {
        "auto_detect_language": True,
        "fallback_language": "en",
        "confidence_threshold": 0.8
    }
}

# Performance Settings
PERFORMANCE_SETTINGS = {
    # Batch processing
    "batch_size": 1,
    
    # Memory management
    "clear_cache_after_processing": True,
    
    # Multiprocessing
    "use_multiprocessing": False,
    "max_workers": 4,
    
    # Progress reporting
    "show_progress": True,
    "verbose_logging": False,
    
    # Quality vs Speed optimization
    "quality_mode": "balanced",  # "fast", "balanced", "quality"
    "enable_gpu_acceleration": True
}

# File Handling Settings
FILE_SETTINGS = {
    # Supported input formats
    "supported_formats": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
    
    # Temporary file handling
    "cleanup_temp_files": True,
    "temp_dir": None,  # None = system default
    
    # Output file naming
    "output_suffix": "_transcription",
    "timestamp_in_filename": False,
    "include_language_in_filename": False,
    
    # Backup settings
    "create_backup": False,
    "backup_dir": "backups"
}

# Validation Settings
VALIDATION_SETTINGS = {
    # Minimum file size (bytes)
    "min_file_size": 1024,
    
    # Maximum file size (bytes) - 500MB
    "max_file_size": 500 * 1024 * 1024,
    
    # Minimum duration (seconds)
    "min_duration": 1.0,
    
    # Maximum duration (seconds) - 2 hours
    "max_duration": 7200,
    
    # Audio quality checks
    "check_audio_quality": True,
    "min_sample_rate": 8000,
    
    # Language validation
    "validate_language": True,
    "language_confidence_threshold": 0.5
}

def get_config_for_use_case(use_case: str = "general") -> dict:
    """
    Get configuration optimized for specific use cases
    
    Args:
        use_case: "general", "support", "sales", "technical", "fast", "accurate", "spanish", "multilingual"
    
    Returns:
        dict: Optimized configuration
    """
    
    configs = {
        "general": {
            "model": "base",
            "language": "auto",
            "speaker_labels": SPEAKER_SETTINGS["alternative_labels"]["support"]
        },
        "support": {
            "model": "base",
            "language": "auto",
            "speaker_labels": SPEAKER_SETTINGS["alternative_labels"]["support"],
            "pause_threshold": 1.5
        },
        "sales": {
            "model": "small",
            "language": "auto",
            "speaker_labels": SPEAKER_SETTINGS["alternative_labels"]["sales"],
            "pause_threshold": 2.5
        },
        "technical": {
            "model": "medium",
            "language": "auto",
            "speaker_labels": SPEAKER_SETTINGS["alternative_labels"]["technical"],
            "pause_threshold": 2.0
        },
        "fast": {
            "model": "tiny",
            "language": "auto",
            "speaker_labels": {"agent": "AGENT", "client": "CLIENT"},
            "pause_threshold": 3.0
        },
        "accurate": {
            "model": "large",
            "language": "auto",
            "speaker_labels": {"agent": "AGENT", "client": "CLIENT"},
            "pause_threshold": 1.0
        },
        "spanish": {
            "model": "medium",
            "language": "es",
            "speaker_labels": SPEAKER_SETTINGS["alternative_labels"]["spanish"],
            "pause_threshold": 1.8,
            "temperature": 0.0
        },
        "spanish_support": {
            "model": "medium",
            "language": "es",
            "speaker_labels": SPEAKER_SETTINGS["alternative_labels"]["support_es"],
            "pause_threshold": 1.5,
            "temperature": 0.0
        },
        "multilingual": {
            "model": "medium",
            "language": "auto",
            "speaker_labels": {"agent": "AGENT", "client": "CLIENT"},
            "pause_threshold": 2.0,
            "auto_detect": True
        }
    }
    
    return configs.get(use_case, configs["general"])

def get_language_specific_config(language: str) -> dict:
    """
    Get language-specific optimizations
    
    Args:
        language: Language code (e.g., 'es', 'en', 'fr')
    
    Returns:
        dict: Language-specific configuration
    """
    
    language_configs = {
        "es": {  # Spanish
            "recommended_model": "medium",
            "pause_threshold": 1.8,
            "speaker_labels": {"agent": "AGENTE", "client": "CLIENTE"},
            "temperature": 0.0,
            "beam_size": 5
        },
        "en": {  # English
            "recommended_model": "base",
            "pause_threshold": 2.0,
            "speaker_labels": {"agent": "AGENT", "client": "CLIENT"},
            "temperature": 0.0,
            "beam_size": 5
        },
        "fr": {  # French
            "recommended_model": "medium",
            "pause_threshold": 1.9,
            "speaker_labels": {"agent": "AGENT", "client": "CLIENT"},
            "temperature": 0.0,
            "beam_size": 5
        },
        "pt": {  # Portuguese
            "recommended_model": "medium",
            "pause_threshold": 1.8,
            "speaker_labels": {"agent": "AGENTE", "client": "CLIENTE"},
            "temperature": 0.0,
            "beam_size": 5
        }
    }
    
    return language_configs.get(language, language_configs["en"])

# Quality Improvement Recommendations
QUALITY_RECOMMENDATIONS = {
    "poor_quality": {
        "suggestions": [
            "Use a larger model (medium or large)",
            "Specify the exact language instead of auto-detection",
            "Check audio quality - ensure clear recording",
            "Preprocess audio to reduce noise",
            "Use WAV format instead of MP3 if possible"
        ],
        "model_upgrade": {
            "tiny": "base",
            "base": "medium",
            "small": "medium",
            "medium": "large"
        }
    },
    "speaker_separation": {
        "suggestions": [
            "Adjust pause_threshold in configuration",
            "Enable advanced speaker diarization",
            "Check if speakers have similar voices",
            "Ensure adequate pause between speakers"
        ]
    },
    "language_detection": {
        "suggestions": [
            "Specify language explicitly",
            "Use larger model for better language detection",
            "Check if audio contains multiple languages",
            "Verify audio quality and clarity"
        ]
    }
}

# Default configuration
DEFAULT_CONFIG = {
    "speaker": SPEAKER_SETTINGS,
    "audio": AUDIO_SETTINGS,
    "whisper": WHISPER_SETTINGS,
    "output": OUTPUT_SETTINGS,
    "ai": AI_SETTINGS,
    "call_center": CALL_CENTER_SETTINGS,
    "performance": PERFORMANCE_SETTINGS,
    "files": FILE_SETTINGS,
    "validation": VALIDATION_SETTINGS,
    "gemini": GEMINI_SETTINGS,
    "quality": QUALITY_RECOMMENDATIONS
}

# Gemini API Helper Functions
def get_gemini_config():
    """Get the Gemini API configuration"""
    return GEMINI_SETTINGS

def validate_gemini_config():
    """Validate that Gemini API is properly configured"""
    api_key = GEMINI_SETTINGS.get("api_key")
    if not api_key or api_key == "xxx":
        return False, "Gemini API key not found or not set. Please set GEMINI_API_KEY in your .env file."
    return True, "Gemini configuration is valid"

def get_gemini_client():
    """
    Get a configured Gemini client
    
    Returns:
        Configured Gemini client or None if not available
    """
    try:
        import google.generativeai as genai
        
        is_valid, message = validate_gemini_config()
        if not is_valid:
            raise ValueError(message)
        
        # Configure the API
        genai.configure(api_key=GEMINI_SETTINGS["api_key"])
        
        # Create model with configuration
        generation_config = GEMINI_SETTINGS["generation_config"]
        
        # Use minimal safety settings for business content
        safety_settings = GEMINI_SETTINGS["safety_settings"]
        
        model = genai.GenerativeModel(
            model_name=GEMINI_SETTINGS["default_model"],
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        return model
        
    except ImportError:
        raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
    except Exception as e:
        raise Exception(f"Failed to initialize Gemini client: {str(e)}")

def create_gemini_prompt(template: str, **kwargs) -> str:
    """
    Create a formatted prompt for Gemini API
    
    Args:
        template: Prompt template with placeholders
        **kwargs: Values to fill in the template
    
    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs) 