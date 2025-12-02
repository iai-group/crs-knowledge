"""Chain creation utilities."""

from typing import Any, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class ChainFactory:
    """Factory for creating LangChain chains."""

    def __init__(self, model, parser: Optional[Any] = None):
        self.model = model
        self.parser = parser or StrOutputParser()

    def create_chain(self, prompt_template: str, parser: Optional[Any] = None):
        """Create a chain with optional custom output parser."""
        prompt = PromptTemplate.from_template(prompt_template)
        output_parser = parser or self.parser
        return prompt | self.model | output_parser


def create_simple_chain(model, prompt_template: str):
    """Legacy function for backward compatibility."""
    factory = ChainFactory(model)
    return factory.create_chain(prompt_template)
