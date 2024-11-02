import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class PodcastConfig:
    """Configuration class that loads settings from .env and yaml files"""
    
    # API Keys
    google_api_key: str
    elevenlabs_api_key: str
    openai_api_key: str
    tavily_api_key: str
    anthropic_api_key: str

    # LLM config
    fast_llm_provider: str
    long_context_llm_provider: str
    
    # TTS Config 
    tts_provider: str
    tts_settings: Dict
    
    # Output Config
    output_format: str
    temp_audio_dir: str
    output_dir: str

    # Rate limits for TTS:
    rate_limits: Dict

    # Checkpointer confif
    checkpoint_dir: str
    
    @classmethod
    def load(cls, yaml_path: Optional[str] = None) -> 'PodcastConfig':
        """
        Load configuration from .env and optional yaml file
        
        Args:
            yaml_path: Optional path to yaml config file
            
        Returns:
            PodcastConfig: Loaded configuration object
        """
        # Load environment variables
        load_dotenv()
        
        # Required API keys from env
        required_env_vars = [
            'GOOGLE_API_KEY',
            'ELEVENLABS_API_KEY', 
            'OPENAI_API_KEY',
            'TAVILY_API_KEY',
            'ANTHROPIC_API_KEY'
        ]
        
        config_dict = {}
        for var in required_env_vars:
            value = os.getenv(var)
            if not value:
                raise ValueError(f'Missing required environment variable: {var}')
            config_dict[var.lower()] = value
            
        # Load and merge yaml config if provided
        if yaml_path:
            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f)
                config_dict.update(yaml_config)
        
        # Set defaults for optional configs
        defaults = {
            'fast_llm_provider': 'openai',
            'long_context_llm_provider': 'openai',
            'tts_provider': 'google',
            'tts_settings': {
                'elevenlabs': {
                    'voice_mapping': {
                        'Interviewer': 'Chris',
                        'Interviewee': 'Charlie'
                    },
                    'model': 'eleven_multilingual_v2'
                },
                'google': {
                    'voice_mapping': {
                        'Interviewer': 'en-US-Journey-F',
                        'Interviewee': 'en-US-Journey-D'
                    },
                    'language_code': 'en-US',
                    'effects_profile_id': 'small-bluetooth-speaker-class-device'
                }
            },
            'output_format': 'mp3',
            'temp_audio_dir': './.temp_audio',
            'output_dir': './output',
            'checkpoint_dir': './.checkpoints',
            'rate_limits': {
                'elevenlabs': {
                    'requests_per_minute': 20,
                    'max_retries': 10,
                    'base_delay': 2.0
                },
                'google': {
                    'requests_per_minute': 20,
                    'max_retries': 10,
                    'base_delay': 2.0
                }
            }
        }
        
        for key, value in defaults.items():
            if key not in config_dict:
                config_dict[key] = value
                
        return cls(**config_dict)
