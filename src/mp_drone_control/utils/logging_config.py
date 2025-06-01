"""Centralized logging configuration for the project."""

import logging
import colorlog
import os

# Set environment variables to suppress MediaPipe warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"  # Suppress glog messages


def setup_logging():
    """Configure logging for the entire project."""
    # Get the root logger
    root_logger = logging.getLogger()

    # Only configure if no handlers are present
    if not root_logger.handlers:
        # Create a formatter that includes colors
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s [%(asctime)s] %(name)s: %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        # Create a handler and set the formatter
        handler = colorlog.StreamHandler()
        handler.setFormatter(formatter)

        # Configure the root logger
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        # Suppress noisy external loggers
        for noisy_logger in [
            "absl",
            "tensorflow",
            "mediapipe",
            "gl_context",
            "inference_feedback_manager",
        ]:
            logging.getLogger(noisy_logger).setLevel(logging.ERROR)

    return root_logger
