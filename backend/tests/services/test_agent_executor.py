"""Tests for agent_executor service."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.agent_executor import (
    build_prompt_with_variables,
    _format_sse,
)


class TestBuildPromptWithVariables:
    def test_simple_replacement(self):
        template = "Hello {{name}}, welcome to {{place}}!"
        result = build_prompt_with_variables(
            template, {"name": "Alice", "place": "Wonderland"}
        )
        assert result == "Hello Alice, welcome to Wonderland!"

    def test_default_value(self):
        template = "Ciao {{name|Utente}}!"
        result = build_prompt_with_variables(template, {})
        assert result == "Ciao Utente!"

    def test_default_value_overridden(self):
        template = "Ciao {{name|Utente}}!"
        result = build_prompt_with_variables(template, {"name": "Marco"})
        assert result == "Ciao Marco!"

    def test_missing_variable_kept(self):
        template = "Value: {{missing}}"
        result = build_prompt_with_variables(template, {})
        assert result == "Value: {{missing}}"

    def test_no_placeholders(self):
        template = "No placeholders here"
        result = build_prompt_with_variables(template, {"unused": "val"})
        assert result == "No placeholders here"

    def test_multiple_same_variable(self):
        template = "{{x}} and {{x}}"
        result = build_prompt_with_variables(template, {"x": "Y"})
        assert result == "Y and Y"

    def test_whitespace_in_variable_name(self):
        template = "{{ name }}"
        result = build_prompt_with_variables(template, {"name": "Bob"})
        assert result == "Bob"

    def test_complex_template(self):
        template = """
System: {{system_instruction}}
Context: {{context}}
Output format: {{format|markdown}}
"""
        result = build_prompt_with_variables(template, {
            "system_instruction": "Be helpful",
            "context": "Document text here",
        })
        assert "Be helpful" in result
        assert "Document text here" in result
        assert "markdown" in result  # default used


class TestFormatSSE:
    def test_format(self):
        result = _format_sse("thought", {"message": "Thinking..."})
        assert result.startswith("event: thought\n")
        assert "Thinking..." in result
        assert result.endswith("\n\n")
