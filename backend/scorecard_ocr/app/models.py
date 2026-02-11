from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScorecardHole(BaseModel):
    hole: int
    par: Optional[int] = None
    handicap_men: Optional[int] = None
    handicap_women: Optional[int] = None
    yardages_by_tee: Dict[str, Optional[int]] = Field(default_factory=dict)


class ScorecardTotals(BaseModel):
    out: Optional[int] = None
    in_total: Optional[int] = Field(default=None, alias="in")
    total: Optional[int] = None


class PlayerHole(BaseModel):
    hole: int
    score: Optional[int] = None
    putts: Optional[int] = None
    fairway: Optional[bool] = None
    green: Optional[bool] = None
    up_down: Optional[bool] = None
    penalties: Optional[int] = None


class ScorecardPlayer(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None
    holes: List[PlayerHole] = Field(default_factory=list)


class ScorecardMetadata(BaseModel):
    tee_boxes: List[str] = Field(default_factory=list)
    rating_men_by_tee: Dict[str, float] = Field(default_factory=dict)
    slope_men_by_tee: Dict[str, int] = Field(default_factory=dict)
    rating_women_by_tee: Dict[str, float] = Field(default_factory=dict)
    slope_women_by_tee: Dict[str, int] = Field(default_factory=dict)


class ScorecardParseResponse(BaseModel):
    confidence: float = 0.0
    holes: List[ScorecardHole] = Field(default_factory=list)
    totals: ScorecardTotals = Field(default_factory=ScorecardTotals)
    flags: List[str] = Field(default_factory=list)
    metadata: ScorecardMetadata = Field(default_factory=ScorecardMetadata)
    player: Optional[ScorecardPlayer] = None
    debug_image_base64: Optional[str] = None
