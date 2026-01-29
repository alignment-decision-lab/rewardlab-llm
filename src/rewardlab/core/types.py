from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Step:
    prompt: str
    response: str

@dataclass
class Trajectory:
    steps: List[Step]
    metadata: Dict

@dataclass
class Reward:
    total: float
    components: Dict[str, float]
