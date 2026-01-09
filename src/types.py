import json

from pydantic import BaseModel


LEFT = 1
UP = 2
RIGHT = 3
DOWN = 4


class Position(BaseModel):
    """ A position on the game board. """
    x: int
    y: int


class PlayerState(BaseModel):
    """ The state of a player in the game. """
    head: Position
    trail: list[Position]
    previous_move: int


class GameState(BaseModel):
    """
    Represents the state of the game.
    This is the main object that you will use!
    """
    board_size: int
    me: PlayerState
    opponent: PlayerState
    board: list[list[int]]

    @classmethod
    def from_json(cls, data: str) -> "GameState":
        """
        Create a GameState object from a JSON string.
        """
        return cls.model_validate_json(data.strip())
