"""
Podcast generation module.

This module provides functionality to generate complete podcast episodes from a given topic.
It handles the end-to-end process including research, script writing, and audio generation.

Example:
    >>> from podcast_llm.generate import generate
    >>> generate(
    ...     topic='Artificial Intelligence',
    ...     qa_rounds=3,
    ...     use_checkpoints=True,
    ...     audio_output='podcast.mp3',
    ...     text_output='script.md',
    ...     config='config.yaml',
    ...     debug=True
    ... )

The generation process includes:
- Background research on the topic
- Outlining the episode structure 
- Writing draft and final scripts
- Converting the script to audio using text-to-speech
- Saving outputs to specified locations

The process can be checkpointed to allow for resuming interrupted generations.
Debug logging provides detailed information about the generation process.
"""


import os
import argparse
from pathlib import Path
from typing import Optional
from podcast_llm.research import (
    research_background_info,
    research_discussion_topics
)
from podcast_llm.writer import (
    write_draft_script,
    write_final_script
)
from podcast_llm.outline import outline_episode
from podcast_llm.utils.checkpointer import (
    Checkpointer,
    to_snake_case
)
from podcast_llm.text_to_speech import generate_audio
from podcast_llm.config import PodcastConfig, setup_logging
from podcast_llm.utils.text import generate_markdown_script
import logging


PACKAGE_ROOT = Path(__file__).parent
DEFAULT_CONFIG_PATH = os.path.join(PACKAGE_ROOT, 'config', 'config.yaml')


def generate(
    topic: str,
    qa_rounds: int,
    use_checkpoints: bool,
    audio_output: str | None,
    text_output: str | None,
    config: str,
    debug: bool
) -> None:
    """
    Generate a podcast episode.
    
    Args:
        topic: Topic of the podcast
        qa_rounds: Number of Q&A rounds
        use_checkpoints: Whether to use checkpointing
        audio_output: Path to save audio output
        text_output: Path to save text output
        config: Path to config file
        debug: Whether to enable debug logging
    """
    # Set logging level based on debug flag
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)
    
    config = PodcastConfig.load(yaml_path=config)

    checkpointer = Checkpointer(
        checkpoint_key=f"{to_snake_case(topic)}_qa_{qa_rounds}_",
        enabled=use_checkpoints
    )

    background_info = checkpointer.checkpoint(research_background_info, [config, topic], stage_name='background_info')
    outline = checkpointer.checkpoint(outline_episode, [config, topic, background_info], stage_name='outline')
    deep_info = checkpointer.checkpoint(research_discussion_topics, [config, topic, outline], stage_name='deep_info')
    draft_script = checkpointer.checkpoint(write_draft_script, [config, topic, outline, background_info, deep_info, qa_rounds], stage_name='draft_script')
    final_script = checkpointer.checkpoint(write_final_script, [config, topic, draft_script], stage_name='final_script')

    if text_output:
        with open(text_output, 'w+') as f:
            f.write(generate_markdown_script(topic, outline, final_script))

    if audio_output:
        generate_audio(config, final_script, audio_output)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate text using a language model'
    )
    parser.add_argument(
        'topic',
        help='Topic of the podcast.'
    )
    parser.add_argument(
        '--qa-rounds',
        type=int,
        default=2,
        help='Number of question-answer rounds per section'
    )
    parser.add_argument(
        '--checkpoint',
        type=bool,
        default=True,
        help='Whether to enable checkpointing'
    )
    parser.add_argument(
        '--audio-output',
        type=str,
        default=None,
        help='Output filename for the generated audio'
    )
    parser.add_argument(
        '--text-output',
        type=str,
        default=None,
        help='Output filename for the generated text script'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    generate(
        topic=args.topic,
        qa_rounds=args.qa_rounds,
        use_checkpoints=args.checkpoint,
        audio_output=args.audio_output,
        text_output=args.text_output,
        config=args.config,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
