"""Lightweight logging bridges for extracted modules still emitting console output."""

from __future__ import annotations

import builtins
import logging
from typing import Any, Callable


def build_print_logger(logger: logging.Logger) -> Callable[..., None]:
    """Mirror legacy print-style output into the module logger."""

    def _emit(*args: Any, **kwargs: Any) -> None:
        builtins.print(*args, **kwargs)
        try:
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            message = sep.join(str(arg) for arg in args)
            if end not in {"", "\n"}:
                message += str(end).rstrip("\n")
            logger.info(message)
        except Exception:
            logger.info(" ".join(str(arg) for arg in args))

    return _emit


__all__ = ["build_print_logger"]
