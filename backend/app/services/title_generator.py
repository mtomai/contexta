from openai import OpenAI
from app.config import get_settings

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)


def generate_conversation_title(first_user_message: str) -> str:
    """
    Generate conversation title from first user message using the light model.
    Falls back to truncated message if API call fails.

    Args:
        first_user_message: First message from user

    Returns:
        Generated title (max 100 chars)
    """
    try:
        response = client.chat.completions.create(
            model=settings.light_model,
            messages=[
                {
                    "role": "user",
                    "content": settings.title_generation_prompt.format(
                        first_message=first_user_message
                    )
                }
            ],
            max_completion_tokens=settings.title_max_tokens
        )

        title = response.choices[0].message.content.strip()
        # Remove quotes if present
        title = title.strip('"\'')
        # Ensure max length
        return title[:100] if title else _fallback_title(first_user_message)

    except Exception as e:
        print(f"Error generating title: {e}")
        # Fallback: truncate first message
        return _fallback_title(first_user_message)


def _fallback_title(message: str) -> str:
    """
    Generate fallback title by truncating message.

    Args:
        message: User message

    Returns:
        Truncated title
    """
    max_length = 50
    if len(message) <= max_length:
        return message
    return message[:max_length] + "..."
