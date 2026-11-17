"""Shared array contracts. Coordinates are normalized unless otherwise stated."""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class TrackingData:
    frame: np.ndarray
    players_xy: np.ndarray
    ball_xy_raw: np.ndarray
    player_observed: np.ndarray
    ball_observed: np.ndarray
    slot_team: np.ndarray
    fps: float = 25.0
    pitch: tuple[float, float] = (105.3, 68.0)
    blocks: list[tuple[int, int]] = field(default_factory=list)

@dataclass
class FeatureData:
    values: dict[str, np.ndarray]
    blocks: list[tuple[int, int]] = field(default_factory=list)
    fps: float = 25.0

@dataclass
class TaggingEvidence:
    team_support: np.ndarray
    restart: np.ndarray
    conflict: np.ndarray

@dataclass
class Evidence:
    """Positive normalized state_scores in H,A,D order; change targets H,A."""
    state_scores: np.ndarray
    change: np.ndarray
    stop: np.ndarray
    restart: np.ndarray

