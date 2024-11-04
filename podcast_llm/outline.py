"""
Podcast outline generation module.

This module provides functionality for generating and structuring podcast outlines.
It defines data models for representing podcast sections and subsections, and includes
utilities for formatting and manipulating outline structures.

The module uses Pydantic models to enforce data validation and provide a clean interface
for working with podcast outline components. The models support hierarchical organization
of content with sections containing subsections, and include helper methods for string
formatting.

Classes:
    PodcastSubsection: Models an individual subsection within a podcast section
    PodcastSection: Models a major section containing multiple subsections

Example:
    section = PodcastSection(
        title="Main Discussion",
        subsections=[
            PodcastSubsection(title="Key Concepts"),
            PodcastSubsection(title="Historical Context")
        ]
    )
    print(section.as_str)
"""


from typing import List
import logging
from langchain import hub
from podcast_llm.config import PodcastConfig
from podcast_llm.utils.llm import get_long_context_llm
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class PodcastSubsection(BaseModel):
    """
    A model representing a subsection within a podcast section.

    This class models an individual subsection of content within a larger podcast section.
    It provides a structured way to store and format subsection titles, with a helper
    property to output the subsection in a standardized string format with appropriate
    indentation.

    Attributes:
        title (str): The title/heading text for this subsection
    """
    title: str = Field(..., description="A subsection in a podcast outline")

    @property
    def as_str(self) -> str:
        return f"-- {self.title}".strip()


class PodcastSection(BaseModel):
    """
    A model representing a section within a podcast outline.

    This class models a major section of content within a podcast outline. Each section
    contains a title and a list of subsections, providing hierarchical structure to
    the podcast content. A helper property formats the section and its subsections
    into a standardized string representation.

    Attributes:
        title (str): The title/heading text for this section
        subsections (List[PodcastSubsection]): List of subsections contained within this section
    """
    title: str = Field(..., description="A section in a podcast outline")
    subsections: List[PodcastSubsection] = Field(..., description="List of subsections in a podcast section")

    @property
    def as_str(self) -> str:
        return f"{self.title}\n{'\n'.join([ss.as_str for ss in self.subsections])}".strip()
    

class PodcastOutline(BaseModel):
    """
    A model representing a complete podcast episode outline.

    This class models the full structure of a podcast episode outline, containing
    an ordered list of major sections, each with their own subsections. It provides
    a hierarchical organization of the episode content and includes a helper property
    to format the entire outline into a standardized string representation.

    Attributes:
        sections (List[PodcastSection]): Ordered list of major sections making up the episode outline
    """
    sections: List[PodcastSection] = Field(..., description="List of sections in a podcast outline")
    
    @property
    def as_str(self) -> str:
        return f"{'\n'.join([s.as_str for s in self.sections])}".strip()


def format_wikipedia_document(doc):
    """
    Format a Wikipedia document for use in prompt context.

    Takes a Wikipedia document object and formats its metadata and content into a 
    structured string format suitable for inclusion in LLM prompts. The format
    includes a header with the article title followed by the full article content.

    Args:
        doc: Wikipedia document object containing metadata and page content

    Returns:
        str: Formatted string with article title and content
    """
    return f"### {doc.metadata['title']}\n\n{doc.page_content}"


def outline_episode(config: PodcastConfig, topic: str, background_info: list) -> PodcastOutline:
    """
    Generate a structured outline for a podcast episode.

    Takes a topic and background research information, then uses LangChain and GPT-4 
    to generate a detailed podcast outline with sections and subsections. The outline
    is structured using Pydantic models for type safety and validation.

    Args:
        topic (str): The main topic for the podcast episode
        background_info (list): List of Wikipedia document objects containing research material

    Returns:
        PodcastOutline: Structured outline object containing sections and subsections
    """
    logger.info(f'Generating outline for podcast on: {topic}')
    
    prompthub_path = "evandempsey/podcast_outline:6ceaa688"
    outline_prompt = hub.pull(prompthub_path, )
    logger.info(f"Got prompt from hub: {prompthub_path}")

    outline_llm = get_long_context_llm(config)
    outline_chain = outline_prompt | outline_llm.with_structured_output(
        PodcastOutline
    )

    outline = outline_chain.invoke({
        "episode_structure": config.episode_structure_for_prompt,
        "topic": topic,
        "context_documents": "\n\n".join([format_wikipedia_document(d) for d in background_info])
    })

    logger.info(outline.as_str)
    return outline
