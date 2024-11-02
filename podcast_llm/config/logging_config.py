import logging
import sys

def setup_logging():
    """
    Set up standardized logging configuration for the podcast generation system.

    Configures a root logger with consistent formatting and stdout output. Handles:
    - Removing any existing handlers to prevent duplicate logs
    - Setting up a StreamHandler to output to stdout
    - Configuring timestamp-based formatting for log messages
    - Setting the default log level to INFO

    The format for log messages is:
    YYYY-MM-DD HH:MM:SS - LEVEL - MESSAGE
    """
    # Remove any existing handlers to avoid duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging to output to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Set up root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
