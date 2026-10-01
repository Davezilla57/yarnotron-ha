from pydantic import BaseModel, Field


class VoiceOption(BaseModel):
    id: str
    name: str
    provider: str
    locale: str | None = None


class ParentPreferences(BaseModel):
    tts_provider: str = "auto"
    tts_voice: str | None = None
    space_friend_howl_enabled: bool = True


class UpdateParentPreferencesRequest(BaseModel):
    tts_provider: str = Field(default="auto", pattern="^(auto|none|kokoro|macos-say|piper)$")
    tts_voice: str | None = None
    space_friend_howl_enabled: bool = True


class VoiceSampleRequest(BaseModel):
    tts_provider: str = Field(default="auto", pattern="^(auto|kokoro|macos-say)$")
    tts_voice: str | None = None
    text: str = Field(
        default="Welcome to Yarnotron, what story would you like to work on?",
        min_length=1,
        max_length=240,
    )
