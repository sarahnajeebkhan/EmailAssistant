from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    email_agent_model: str | None = None
    email_nvidia_api_key: str | None = None

    # Google credentials configuration
    google_credentials_path: str = "credentials.json"
    google_token_path: str = "token.json"
    google_api_key: str | None = None
    google_cse_id: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="EMAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_model(self):
        """Get the configured model or raise error if NVIDIA key is missing."""
        if self.email_agent_model is None:
            if not self.email_nvidia_api_key:
                raise ValueError(
                    "EMAIL_NVIDIA_API_KEY environment variable is required. "
                    "Set it or override EMAIL_AGENT_MODEL to use a different LLM."
                )
            return self._create_nvidia_model()
        return self.email_agent_model

    def _create_nvidia_model(self):
        """Create NVIDIA ChatNVIDIA model instance."""
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        return ChatNVIDIA(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key=self.email_nvidia_api_key,
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            reasoning_budget=16384,
            chat_template_kwargs={"enable_thinking": True},
        )


settings = Settings()
