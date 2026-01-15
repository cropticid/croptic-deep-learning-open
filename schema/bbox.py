from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Koordinat kiri atas X")
    y1: float = Field(..., description="Koordinat kiri atas Y")
    x2: float = Field(..., description="Koordinat kanan bawah X")
    y2: float = Field(..., description="Koordinat kanan bawah Y")
    label: Optional[str] = Field(None, description="Label objek (opsional)")

class BBoxListInput(BaseModel):
    bboxes: List[BoundingBox]