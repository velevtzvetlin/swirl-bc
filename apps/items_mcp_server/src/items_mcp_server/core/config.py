from pydantic_settings import BaseSettings, SettingsConfigDict

# this will always require the three keys in the environment variables otherwise this schema would error out
class Config(BaseSettings):
    OPENAI_API_KEY: str
    CO_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env")

config = Config() ## instantiate the config class...