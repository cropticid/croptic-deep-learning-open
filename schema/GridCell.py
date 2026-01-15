from pydantic import BaseModel
from typing import Dict

class GridCell(BaseModel):
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int
    greenness: float = 0.0
    features: Dict = None