import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

class SafEvalLogger:
    """
    Logger class for SAF-Eval pipeline that supports both console and file logging,
    with structured data output in JSON format for better analysis.
    """
    
    def __init__(
        self, 
        name: str = "saf-eval", 
        level: int = logging.INFO,
        log_dir: Optional[str] = None,
        console: bool = True,
        file: bool = False,
        json_format: bool = False
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level (default: INFO)
            log_dir: Directory for log files (default: ./logs)
            console: Whether to log to console
            file: Whether to log to file
            json_format: Whether to format logs as JSON
        """
        self.name = name
        self.level = level
        self.json_format = json_format
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            if json_format:
                console_handler.setFormatter(logging.Formatter('%(message)s'))
            else:
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            self.logger.addHandler(console_handler)
        
        # File handler
        if file:
            if log_dir is None:
                log_dir = os.path.join(os.getcwd(), 'logs')
            
            # Create log directory if it doesn't exist
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            
            # Create log filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f"{name}_{timestamp}.log"
            log_path = os.path.join(log_dir, log_filename)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            if json_format:
                file_handler.setFormatter(logging.Formatter('%(message)s'))
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            self.logger.addHandler(file_handler)
    
    def _format_message(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Format message, optionally as JSON with extra fields."""
        if not self.json_format or extra is None:
            return msg
            
        log_data = {"message": msg, **extra}
        return json.dumps(log_data)
    
    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(msg, extra))
    
    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        self.logger.info(self._format_message(msg, extra))
    
    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(msg, extra))
    
    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message."""
        self.logger.error(self._format_message(msg, extra))
    
    def critical(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(msg, extra))

# Create a default logger instance
default_logger = SafEvalLogger()

def get_logger(
    name: str = "saf-eval",
    level: Union[int, str] = logging.INFO,
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = False,
    json_format: bool = False
) -> SafEvalLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_dir: Directory for log files (default: ./logs)
        console: Whether to log to console
        file: Whether to log to file
        json_format: Whether to format logs as JSON
        
    Returns:
        Configured SafEvalLogger instance
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
        
    return SafEvalLogger(name, level, log_dir, console, file, json_format)
