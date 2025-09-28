"""
Utility Functions for SWE Training Framework.

This module provides core utility functions used throughout the software engineering
training pipeline. It includes message processing, tokenization utilities, and
helper functions for managing chat completions and conversation flows.

Key Components:
- get_recent_assistant_user_messages: Extract recent conversation context
- convert_messages_to_tokens_and_masks: Convert chat messages to model inputs
- Message parsing and formatting utilities
- Token and attention mask generation for training

The utilities support various chat template formats and provide consistent
interfaces for processing conversational data in the distributed training
workflow. They handle special tokens, attention masks, and position encoding
for different model architectures.

Features:
- Multi-turn conversation processing
- Flexible tokenization with custom chat templates
- Attention mask generation for training efficiency
- Support for tool calls and system messages
- Integration with various tokenizer formats

Usage:
    from recipe.qianfan_swe.trainer.utils.utils import (
        get_recent_assistant_user_messages,
        convert_messages_to_tokens_and_masks
    )
    
    # Extract recent conversation context
    assistant_msg, env_msgs = get_recent_assistant_user_messages(messages)
    
    # Convert to tokens for model input
    tokens, masks = convert_messages_to_tokens_and_masks(
        messages, tokenizer, parser
    )

The module is designed to work seamlessly with the training framework's
distributed architecture and supports both SWE-bench and R2E evaluation
environments.
"""


from transformers import PreTrainedTokenizerBase
from .parser import ChatTemplateParser


def convert_messages_to_tokens_and_masks(messages: list[dict[str, str]], tokenizer: PreTrainedTokenizerBase, parser: ChatTemplateParser, contains_first_msg=False, contains_generation_msg=False):
    """
    Converts multiple messages to tokens and masks.
    contains_first_msg flag and contains_generaiton_msg flag are used to indicate whether the conversation is for beginning or contains the generation.
    The first and last message is assumed to be the special message respectively

    Args:
        messages (List[Dict]): The messages to convert.
        tokenizer: The tokenizer to use.
        contains_first_msg (bool): Whether the first message is a special message.
        contains_generation_msg (bool): Whether the last message is a special message.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing all tokens and all masks.
    """
    all_msg_tokens = []
    all_msg_masks = []

    def _convert_message_to_tokens_and_masks(msg, first_msg=False, generation_msg=False):
        msg_text = parser.parse([msg], add_generation_prompt=generation_msg, is_first_msg=first_msg)

        # Remove the assistant token since it is contained in previous message as generation prompt
        if msg["role"] == "assistant":
            assert msg_text.startswith(parser.assistant_token), f"Expected assistant token {parser.assistant_token} but got {msg_text}"
            msg_text = msg_text.replace(parser.assistant_token, "")

        msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        mask_value = 1 if msg["role"] == "assistant" else 0
        msg_mask = [mask_value] * len(msg_tokens)

        return msg_tokens, msg_mask

    for i, msg in enumerate(messages):
        msg_tokens, msg_mask = _convert_message_to_tokens_and_masks(msg, first_msg=(contains_first_msg and i == 0), generation_msg=(contains_generation_msg and i == len(messages) - 1))
        all_msg_tokens.extend(msg_tokens)
        all_msg_masks.extend(msg_mask)

    return all_msg_tokens, all_msg_masks