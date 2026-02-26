"""Customer agent for simulating realistic user interactions."""

import os
import time
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class CustomerAgent:
    """Simulates realistic customer responses using OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        rate_limit_delay: float = 1.0,
        temperature: float = 0.01
    ):
        """Initialize customer agent.

        Args:
            api_key: OpenAI API key (reads from env if not provided)
            model: Model name to use
            rate_limit_delay: Delay between API calls in seconds
            temperature: Temperature for generation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env file "
                "or pass it to the constructor."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.rate_limit_delay = rate_limit_delay
        self.temperature = temperature

        logger.info(f"Initialized CustomerAgent with model: {model}")

    def generate_response(
        self,
        conversation_history: str,
        chatbot_question: str
    ) -> str:
        """Generate a realistic customer response.

        Args:
            conversation_history: Full conversation history so far
            chatbot_question: Current question from the chatbot

        Returns:
            Generated customer response
        """
        system_prompt = f"""
You are playing the role of a real human user chatting with a support chatbot.

This is your conversation so far:
{conversation_history}

Now the chatbot is trying to understand your intent. It may ask you to choose 
between two or three possible options, or clarify your issue.

Respond like a real person would. Sometimes people are clear and direct and may 
choose an option they find reasonable, sometimes they're informal or unsure. 
Use your own words.

It's okay to be casual, make typos, or ramble a bit sometimes. Vary your 
behavior the way real users do.

Do not keep repeating the same question again and again. Behave like a user 
who wants to make the chatbot understand their query.

If the chatbot responds with something completely irrelevant to your question, 
inform it so.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chatbot_question}
        ]

        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )

            response_text = response.choices[0].message.content.strip()
            logger.debug(f"Generated response: {response_text}")
            return response_text

        except Exception as e:
            logger.error(f"Error generating customer response: {e}")
            raise

    def __call__(
        self,
        conversation_history: str,
        chatbot_question: str
    ) -> str:
        """Make the agent callable.

        Args:
            conversation_history: Full conversation history
            chatbot_question: Current chatbot question

        Returns:
            Generated response
        """
        return self.generate_response(conversation_history, chatbot_question)


def customer_agent(
    conversation_history: str,
    chatbot_question: str,
    api_key: Optional[str] = None
) -> str:
    """Convenience function for backward compatibility.

    Args:
        conversation_history: Full conversation history
        chatbot_question: Current chatbot question
        api_key: Optional API key

    Returns:
        Generated customer response
    """
    agent = CustomerAgent(api_key=api_key)
    return agent.generate_response(conversation_history, chatbot_question)
