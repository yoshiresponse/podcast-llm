from typing import Any, List
import logging
from langchain import hub
from podcast_llm.config import PodcastConfig
from podcast_llm.utils.llm import get_long_context_llm

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class PodcastSubsection(BaseModel):
    title: str = Field(..., description="A subsection in a podcast outline")

    @property
    def as_str(self) -> str:
        return f"-- {self.title}".strip()


class PodcastSection(BaseModel):
    title: str = Field(..., description="A section in a podcast outline")
    subsections: List[PodcastSubsection] = Field(..., description="List of subsections in a podcast section")

    @property
    def as_str(self) -> str:
        return f"{self.title}\n{'\n'.join([ss.as_str for ss in self.subsections])}".strip()
    

class PodcastOutline(BaseModel):
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
    
    prompthub_path = "evandempsey/podcast_outline"
    outline_prompt = hub.pull(prompthub_path)
    logger.info(f"Got prompt from hub: {prompthub_path}")

    outline_llm = get_long_context_llm(config)
    outline_chain = outline_prompt | outline_llm.with_structured_output(
        PodcastOutline
    )

    outline = outline_chain.invoke({
        "topic": topic,
        "context_documents": "\n\n".join([format_wikipedia_document(d) for d in background_info])
    })

    logger.info(outline.as_str)
    return outline
