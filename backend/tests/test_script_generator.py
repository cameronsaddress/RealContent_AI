"""
Tests for ScriptGenerator service - uses mocked responses, no real API calls.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from services.script_generator import (
    ScriptGenerator,
    ScriptRequest,
    ScriptScene,
    GeneratedScript
)


class TestScriptRequest:
    """Test ScriptRequest model."""

    def test_defaults(self):
        """Test default values."""
        request = ScriptRequest(content_idea_id=123)
        assert request.content_idea_id == 123
        assert request.pillar == "educational_tips"
        assert request.target_duration == 60
        assert request.location == "Coeur d'Alene, Idaho"
        assert request.character_name == "Beth"

    def test_custom_values(self):
        """Test custom values."""
        request = ScriptRequest(
            content_idea_id=456,
            pillar="market_intelligence",
            source_url="https://tiktok.com/video/123",
            original_text="Original viral content",
            suggested_hook="You won't believe this market data",
            target_duration=45,
            location="Spokane, Washington",
            character_name="Sarah"
        )
        assert request.pillar == "market_intelligence"
        assert request.target_duration == 45
        assert request.character_name == "Sarah"


class TestScriptScene:
    """Test ScriptScene model."""

    def test_minimal_scene(self):
        """Test with minimal fields."""
        scene = ScriptScene(
            scene_number=1,
            text="Opening shot",
            duration_estimate=5.0
        )
        assert scene.scene_number == 1
        assert scene.broll_keywords == []
        assert scene.broll_description == ""

    def test_full_scene(self):
        """Test with all fields."""
        scene = ScriptScene(
            scene_number=2,
            text="Showing the beautiful home",
            duration_estimate=8.5,
            broll_keywords=["luxury home", "modern kitchen"],
            broll_description="Wide shot of modern luxury home exterior"
        )
        assert scene.duration_estimate == 8.5
        assert len(scene.broll_keywords) == 2


class TestGeneratedScript:
    """Test GeneratedScript model."""

    def test_full_script(self):
        """Test complete generated script."""
        script = GeneratedScript(
            content_idea_id=123,
            hook="Want to know the secret to finding the perfect home?",
            body="Here are three tips that changed everything for my clients.",
            cta="DM me 'TIPS' for your personalized home search plan!",
            full_text="Want to know... Here are... DM me...",
            duration_estimate=45.5,
            scenes=[
                ScriptScene(scene_number=1, text="Hook", duration_estimate=5.0),
                ScriptScene(scene_number=2, text="Tips", duration_estimate=30.0)
            ],
            pillar="educational_tips",
            metadata={"model": "grok-4.1-fast"}
        )
        assert len(script.scenes) == 2
        assert script.duration_estimate == 45.5


class TestScriptGenerator:
    """Test ScriptGenerator methods."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return ScriptGenerator()

    def test_build_prompt_educational(self, generator):
        """Test prompt building for educational pillar."""
        request = ScriptRequest(
            content_idea_id=123,
            pillar="educational_tips",
            source_url="https://tiktok.com/test",
            original_text="How to save for a home",
            suggested_hook="Here's what nobody tells you",
            target_duration=60
        )
        prompt = generator.build_prompt(request)

        assert "educational_tips" in prompt
        assert "actionable advice" in prompt
        assert "Beth" in prompt
        assert "Coeur d'Alene" in prompt
        assert "60 seconds" in prompt
        assert "tiktok.com" in prompt

    def test_build_prompt_market_intelligence(self, generator):
        """Test prompt building for market intelligence pillar."""
        request = ScriptRequest(
            content_idea_id=123,
            pillar="market_intelligence"
        )
        prompt = generator.build_prompt(request)

        assert "market_intelligence" in prompt
        assert "data-driven" in prompt.lower() or "statistics" in prompt.lower()

    def test_build_prompt_lifestyle_local(self, generator):
        """Test prompt building for lifestyle local pillar."""
        request = ScriptRequest(
            content_idea_id=123,
            pillar="lifestyle_local",
            location="Liberty Lake, Washington"
        )
        prompt = generator.build_prompt(request)

        assert "lifestyle_local" in prompt
        assert "community" in prompt.lower()
        assert "Liberty Lake" in prompt

    def test_build_prompt_brand_humanization(self, generator):
        """Test prompt building for brand humanization pillar."""
        request = ScriptRequest(
            content_idea_id=123,
            pillar="brand_humanization"
        )
        prompt = generator.build_prompt(request)

        assert "brand_humanization" in prompt
        assert "personal" in prompt.lower() or "authentic" in prompt.lower()

    def test_parse_response(self, generator):
        """Test parsing LLM response."""
        raw = {
            "hook": "You won't believe what happened in the market today!",
            "body": "Here's the breakdown: First, interest rates moved. Second, inventory changed. Third, prices adjusted.",
            "cta": "DM me 'MARKET' for your personalized analysis!",
            "duration_estimate": 52.5,
            "scenes": [
                {
                    "scene_number": 1,
                    "text": "Hook delivery",
                    "duration_estimate": 5.0,
                    "broll_keywords": ["market graph", "news ticker"],
                    "broll_description": "Financial graphics showing market trends"
                },
                {
                    "scene_number": 2,
                    "text": "Main content",
                    "duration_estimate": 40.0,
                    "broll_keywords": ["house exterior", "for sale sign"],
                    "broll_description": "Real estate imagery"
                }
            ]
        }
        request = ScriptRequest(content_idea_id=123, pillar="market_intelligence")

        result = generator.parse_response(raw, request)

        assert result.content_idea_id == 123
        assert "market today" in result.hook
        assert "breakdown" in result.body
        assert "DM me" in result.cta
        assert result.duration_estimate == 52.5
        assert len(result.scenes) == 2
        assert result.scenes[0].broll_keywords == ["market graph", "news ticker"]
        assert result.pillar == "market_intelligence"
        assert "generated_at" in result.metadata

    def test_parse_response_minimal(self, generator):
        """Test parsing minimal LLM response."""
        raw = {
            "hook": "Quick tip!",
            "body": "Here it is.",
            "cta": "Follow me!"
        }
        request = ScriptRequest(content_idea_id=456)

        result = generator.parse_response(raw, request)

        assert result.hook == "Quick tip!"
        assert result.full_text == "Quick tip! Here it is. Follow me!"
        assert result.duration_estimate == 45.0  # Default
        assert len(result.scenes) == 0

    def test_estimate_duration(self, generator):
        """Test duration estimation."""
        # 150 words at 150 wpm = 60 seconds
        text = " ".join(["word"] * 150)
        duration = generator.estimate_duration(text, words_per_minute=150)
        assert duration == 60.0

        # 75 words at 150 wpm = 30 seconds
        text = " ".join(["word"] * 75)
        duration = generator.estimate_duration(text, words_per_minute=150)
        assert duration == 30.0

    def test_format_for_tts(self, generator):
        """Test TTS formatting."""
        script = GeneratedScript(
            content_idea_id=123,
            hook="Hey everyone!",
            body="Here's some great content.",
            cta="Follow for more!",
            full_text="",
            duration_estimate=30.0,
            pillar="educational_tips"
        )

        formatted = generator.format_for_tts(script)

        assert "Hey everyone!" in formatted
        assert "..." in formatted  # Pauses added
        assert "Follow for more!" in formatted

    def test_extract_broll_keywords(self, generator):
        """Test B-roll keyword extraction."""
        script = GeneratedScript(
            content_idea_id=123,
            hook="",
            body="",
            cta="",
            full_text="",
            duration_estimate=30.0,
            scenes=[
                ScriptScene(
                    scene_number=1,
                    text="Scene 1",
                    duration_estimate=10.0,
                    broll_keywords=["house", "garden"]
                ),
                ScriptScene(
                    scene_number=2,
                    text="Scene 2",
                    duration_estimate=10.0,
                    broll_keywords=["kitchen", "house"]  # "house" is duplicate
                )
            ],
            pillar="educational_tips"
        )

        keywords = generator.extract_broll_keywords(script)

        assert len(keywords) == 3  # Deduplicated
        assert "house" in keywords
        assert "garden" in keywords
        assert "kitchen" in keywords


class TestScriptGeneratorAsync:
    """Test async methods with mocked HTTP responses."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return ScriptGenerator()

    @pytest.mark.asyncio
    async def test_generate_script_mocked(self, generator):
        """Test script generation with mocked LLM response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "hook": "Did you know this about Coeur d'Alene?",
                        "body": "The market is changing rapidly. Here's what you need to know about local trends.",
                        "cta": "DM me 'CDA' for your free market report!",
                        "duration_estimate": 48.0,
                        "scenes": [
                            {
                                "scene_number": 1,
                                "text": "Hook",
                                "duration_estimate": 5.0,
                                "broll_keywords": ["Coeur d'Alene lake", "downtown"],
                                "broll_description": "Beautiful lake and city shots"
                            },
                            {
                                "scene_number": 2,
                                "text": "Market data",
                                "duration_estimate": 35.0,
                                "broll_keywords": ["homes", "for sale sign"],
                                "broll_description": "Real estate montage"
                            }
                        ]
                    })
                }
            }]
        }

        with patch.object(generator, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            request = ScriptRequest(
                content_idea_id=141,
                pillar="market_intelligence",
                source_url="https://tiktok.com/test"
            )
            result = await generator.generate_script(request)

            assert result.content_idea_id == 141
            assert "Coeur d'Alene" in result.hook
            assert result.duration_estimate == 48.0
            assert len(result.scenes) == 2
            assert result.pillar == "market_intelligence"

            # Verify OpenRouter was called
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "openrouter.ai" in call_args[0][0]
            assert "grok-4.1-fast" in str(call_args[1]["json"])

    @pytest.mark.asyncio
    async def test_generate_script_error_handling(self, generator):
        """Test that generation errors are raised."""
        with patch.object(generator, '_post', new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("API timeout")

            request = ScriptRequest(content_idea_id=123)

            with pytest.raises(Exception) as exc_info:
                await generator.generate_script(request)

            assert "API timeout" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
