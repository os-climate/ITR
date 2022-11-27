from __future__ import annotations

import typing
import logging

# This lonely class is all by itself because the circular dependencies between Configs and Interfaces
# prevent a proper disposition in either, if we want to be able to emit logging messages from both.
# Patches to resolve are welcome!

class LoggingConfig:
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @classmethod
    def add_config_to_logger(cls, logger: logging.Logger):
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(cls.FORMAT)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)

