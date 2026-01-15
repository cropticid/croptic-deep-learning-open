from pydantic import BaseModel, Field
from typing import List, Dict

class ClusterPolygon(BaseModel):
    polygon: List[List[float]] = Field(..., description="List koordinat [x, y] polygon convex hull pada canvas")
    center: List[float] = Field(..., description="Pusat (median) cluster, untuk label cluster")
    count: int = Field(..., description="Jumlah anggota cluster")