"""Utility functions for working with embeddings models.

This module provides functionality for loading and managing embeddings models,
which are used to convert text into vector representations. Currently supports
Google embeddings.

Functions:
    get_embeddings_model: Returns an initialized embeddings model based on config.
"""

import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from podcast_llm.config import PodcastConfig

logger = logging.getLogger(__name__)

def get_embeddings_model(config: PodcastConfig):
    """Get the configured embeddings model instance.

    Args:
        config (PodcastConfig): Configuration object containing embeddings settings

    Returns:
        BaseEmbeddings: Initialized embeddings model instance based on config.embeddings_model.
            Currently supports 'google' which returns GoogleGenerativeAIEmbeddings.
            Defaults to GoogleGenerativeAIEmbeddings with model="text-embedding-004" if model type is not recognized.
    """
    models = {
        'google': GoogleGenerativeAIEmbeddings
    }

    # Provide the required "model" parameter. Adjust the default as needed.
    return models.get(config.embeddings_model, GoogleGenerativeAIEmbeddings)(model="models/text-embedding-004")
