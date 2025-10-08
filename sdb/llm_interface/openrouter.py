"""OpenRouter API client implementation."""

import os
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional

from sdb.llm_interface.base import BaseLLMClient, LLMResponse
from sdb.llm_interface.utils import retry_with_backoff
from sdb.core.exceptions import LLMError


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API (supports multiple LLM providers)."""
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (or from OPENROUTER_API_KEY env var)
            model: Model identifier (e.g., "openai/gpt-5", "anthropic/claude-opus-4.1")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError("OpenRouter API key not provided")
        
        super().__init__(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    @retry_with_backoff(max_retries=3)
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Make a chat completion API call to OpenRouter.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
            
        Raises:
            LLMError: If API call fails
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.API_URL,
                    headers=self.headers,
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise LLMError(
                            f"OpenRouter API error (status {resp.status})",
                            details={"error": error_text, "model": self.model}
                        )
                    
                    data = await resp.json()
                    
                    # Check for API errors in response
                    if "error" in data:
                        error_msg = data["error"].get("message", str(data["error"]))
                        raise LLMError(
                            f"API returned error: {error_msg}",
                            details={"error": data["error"], "model": self.model}
                        )
                    
                    # Parse response
                    choices = data.get("choices", [])
                    
                    if not choices:
                        # Log the full response for debugging
                        print(f"   ⚠️  WARNING: Empty choices array!")
                        print(f"   Model: {self.model}")
                        print(f"   Full response: {data}")
                        raise LLMError(
                            "No choices returned from API",
                            details={"response": data, "model": self.model}
                        )
                    
                    choice = choices[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    finish_reason = choice.get("finish_reason")
                    
                    # Token usage
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                    
                    response = LLMResponse(
                        content=content,
                        model=self.model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        finish_reason=finish_reason,
                        metadata={"raw_response": data}
                    )
                    
                    self._update_stats(response)
                    return response
                    
        except aiohttp.ClientError as e:
            raise LLMError(f"Network error: {str(e)}", details={"model": self.model})
        except Exception as e:
            raise LLMError(f"Unexpected error: {str(e)}", details={"model": self.model})

