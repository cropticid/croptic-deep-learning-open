from typing import List, Optional
from pydantic import BaseModel, Field

class Polygon(BaseModel):
    points: List[List[float]] = Field(..., description="List of [x, y] points defining the polygon")
    label: Optional[str] = Field(None, description="Label untuk objek segmen (kelas)")

class PolygonListInput(BaseModel):
    polygons: List[Polygon]
