"""
Event extraction pipeline.

Responsible for converting raw user dialogue into
structured event frames used by the update phase.
"""

from core.data_structures import EventFrame


def extract_event(user_message: str) -> EventFrame:
    """
    Convert raw dialogue into a structured event frame.

    Preconditions
    -------------
    user_message : str

    Procedure
    ---------
    1. Send message to LLM extraction prompt
    2. Parse structured JSON output
    3. Construct EventFrame

    Postconditions
    --------------
    Returns EventFrame representing the dialogue event
    """

    # INCOMPLETE

    pass


def validate_event(event: EventFrame, user_message: str) -> EventFrame:
    """
    Validate event extraction.

    Preconditions
    -------------
    event : EventFrame
    user_message : str

    Procedure
    ---------
    1. Check schema validity
    2. Verify alignment with source text
    3. Adjust confidence score

    Postconditions
    --------------
    Returns corrected EventFrame
    """

    # INCOMPLETE

    pass