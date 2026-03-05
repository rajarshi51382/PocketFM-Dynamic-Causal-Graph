"""
Dialogue generation module.

Responsible for generating responses conditioned
on character state.
"""

from core.data_structures import CharacterState


def build_generation_prompt(state: CharacterState, user_message: str) -> str:
    """
    Construct an LLM prompt using the character state.

    Preconditions
    -------------
    state : CharacterState
    user_message : str

    Procedure
    ---------
    Extract relevant beliefs, emotions, and intentions
    Format them into a prompt template

    Postconditions
    --------------
    Returns prompt string
    """

    # INCOMPLETE

    pass


def generate_response(prompt: str) -> str:
    """
    Generate dialogue using the LLM.

    Preconditions
    -------------
    prompt : str

    Procedure
    ---------
    Send prompt to language model.

    Postconditions
    --------------
    Returns generated response text
    """

    # INCOMPLETE

    pass


def produce_dialogue(state: CharacterState, user_message: str) -> str:
    """
    Main generation pipeline.

    Preconditions
    -------------
    state : CharacterState
    user_message : str

    Procedure
    ---------
    Build prompt
    Generate response

    Postconditions
    --------------
    Returns dialogue response
    """

    # INCOMPLETE

    pass