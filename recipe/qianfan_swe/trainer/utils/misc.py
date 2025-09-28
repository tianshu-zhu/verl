"""
Miscellaneous Utilities for SWE Training Framework.

This module provides general-purpose utility functions used throughout the software
engineering training pipeline. It includes formatting utilities, logging helpers,
and common operations that support the distributed training workflow.

Key Components:
- colorful_print: Terminal output formatting with ANSI color codes
- Text formatting and display utilities
- Common helper functions for training operations

The utilities in this module are designed to be lightweight and reusable across
different components of the training framework, providing consistent formatting
and logging capabilities for improved debugging and monitoring.

Features:
- ANSI color code support for terminal output
- Flexible text formatting options
- Cross-platform compatibility
- Integration with logging systems
- Performance-optimized implementations

Usage:
    from recipe.qianfan_swe.trainer.utils.misc import colorful_print
    
    colorful_print("Success!", fg="green", bold=True)
    colorful_print("Warning", fg="yellow", bg="red")
    colorful_print("Info message", fg="blue")

The module supports standard terminal colors and formatting options, making it
easy to create visually distinct output for different types of messages and
status updates throughout the training process.
"""

from typing import Optional


def colorful_print(text: str, fg: Optional[str] = None, bg: Optional[str] = None, bold: bool = False, end: str = "\n"):
    """
    Print colored text to terminal using ANSI color codes
    
    Args:
        text: Text to print
        fg: Foreground color (red, green, yellow, blue, magenta, cyan, white)
        bg: Background color (same options as fg)
        bold: Whether to make text bold
        end: String appended after the last value, default a newline
    """
    # ANSI color codes
    colors = {
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 37,
    }
    
    # Build ANSI escape sequence
    codes = []
    
    if bold:
        codes.append('1')
    
    if fg and fg in colors:
        codes.append(str(colors[fg]))
    
    if bg and bg in colors:
        codes.append(str(colors[bg] + 10))  # Background colors are +10
    
    if codes:
        color_code = '\033[' + ';'.join(codes) + 'm'
        reset_code = '\033[0m'
        print(f"{color_code}{text}{reset_code}", end=end)
    else:
        print(text, end=end)
