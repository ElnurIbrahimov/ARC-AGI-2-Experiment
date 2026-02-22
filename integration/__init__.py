"""
Integration layer for ARC-AGI-2.

Adapter modules that bridge existing research components (Causeway, BroadMind,
FluxMind) to the ARC domain with proper dimensional mapping.

Phase 1: Individual adapters (causeway_adapter, broadmind_adapter, fluxmind_adapter)
Phase 2: Orchestrator (causal_program_bridge) composing all three
"""

from integration.causeway_adapter import CausewayAdapter, ARCDelta
from integration.broadmind_adapter import BroadMindAdapter, BroadMindResult
from integration.fluxmind_adapter import FluxMindAdapter
from integration.causal_program_bridge import (
    CausalProgramBridge,
    ArcExecutionResult,
    WisdomFusionGate,
    ScoreFusionNetwork,
    CausalToWisdomBridge,
    build_causal_program_bridge,
)

__all__ = [
    # Phase 1: Adapters
    "CausewayAdapter",
    "ARCDelta",
    "BroadMindAdapter",
    "BroadMindResult",
    "FluxMindAdapter",
    # Phase 2: Orchestrator
    "CausalProgramBridge",
    "ArcExecutionResult",
    "WisdomFusionGate",
    "ScoreFusionNetwork",
    "CausalToWisdomBridge",
    "build_causal_program_bridge",
]
