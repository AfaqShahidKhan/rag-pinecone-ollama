"""
src/factories/sdk_client_factory.py

The single place in the codebase allowed to construct raw third-party SDK
client objects (pinecone.Pinecone, ollama.Client). Adapters receive these
clients via constructor injection — they never build their own.
"""

from __future__ import annotations

from ollama import Client as OllamaClient
from pinecone import Pinecone

from src.config.settings import OllamaSettings, PineconeSettings


class SdkClientFactory:
    @staticmethod
    def create_pinecone_client(settings: PineconeSettings) -> Pinecone:
        return Pinecone(api_key=settings.api_key)

    @staticmethod
    def create_ollama_client(settings: OllamaSettings) -> OllamaClient:
        return OllamaClient(host=settings.base_url)
