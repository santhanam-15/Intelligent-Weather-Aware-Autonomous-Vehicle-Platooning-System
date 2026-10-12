# WindowsNoEditor/Module3/platoon_coordinator.py
"""Multi-vehicle coordination engine for emergency platoon evasion and rejoining.

Prevents platoon members from blindly copying evasive maneuvers, isolates detached
vehicles, recomputes safe following targets, and coordinates safe formation re-entry.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from Module3.snapshot import WorldSnapshot, VehicleSnapshot
from Module3.evasion_state_machine import EvasionState, VehicleEvasionTracker

logger = logging.getLogger("PlatoonEvasionCoordinator")


class PlatoonVehicleCoordinationState:
    """Tracking structure for an individual vehicle within coordinated platoon operations."""
    __slots__ = (
        "vehicle_id", "role", "is_evading", "evasion_state", "lateral_offset_m",
        "effective_predecessor_id", "desired_gap_override_m"
    )

    def __init__(
        self,
        vehicle_id: int,
        role: str,
        is_evading: bool = False,
        evasion_state: EvasionState = EvasionState.NORMAL,
        lateral_offset_m: float = 0.0,
        effective_predecessor_id: Optional[int] = None,
        desired_gap_override_m: Optional[float] = None,
    ):
        self.vehicle_id = int(vehicle_id)
        self.role = str(role)
        self.is_evading = bool(is_evading)
        self.evasion_state = evasion_state
        self.lateral_offset_m = float(lateral_offset_m)
        self.effective_predecessor_id = effective_predecessor_id
        self.desired_gap_override_m = desired_gap_override_m


class PlatoonEvasionCoordinator:
    """Centralized coordinator ensuring formation integrity during unilateral or multi-vehicle evasions."""

    def __init__(self):
        self.coordination_states: Dict[int, PlatoonVehicleCoordinationState] = {}

    def coordinate_platoon(
        self,
        snapshot: WorldSnapshot,
        trackers: Dict[int, VehicleEvasionTracker],
        tactical_actions: List[str],
    ) -> Tuple[List[str], Dict[int, PlatoonVehicleCoordinationState]]:
        """Filter tactical actions to prevent follower copycatting and adjust following targets.
        
        Returns:
            coordinated_actions: List of sanitized tactical actions per platoon member
            coord_states: Dict mapping vehicle_id to coordination state
        """
        coordinated = list(tactical_actions)
        states: Dict[int, PlatoonVehicleCoordinationState] = {}
        num_veh = len(snapshot.platoon)

        active_evaders: Set[int] = set()
        for v in snapshot.platoon:
            tr = trackers.get(v.id)
            if tr and tr.state not in (EvasionState.NORMAL, EvasionState.RECOVERY):
                active_evaders.add(v.id)

        # 1. Update individual coordination states
        for idx, veh in enumerate(snapshot.platoon):
            tr = trackers.get(veh.id)
            is_evading = veh.id in active_evaders
            ev_state = tr.state if tr else EvasionState.NORMAL
            lat_off = tr.target_lateral_offset_m if tr else 0.0

            # Determine effective predecessor:
            # If the immediate predecessor is currently detached/evading in an adjacent lane,
            # bypass them and follow the nearest non-evading vehicle ahead in the platoon lane!
            eff_pred_id = None
            gap_override = None

            if idx > 0:
                # Look upstream for the first non-evading predecessor
                found_pred = None
                for up_idx in range(idx - 1, -1, -1):
                    cand_pred = snapshot.platoon[up_idx]
                    if cand_pred.id not in active_evaders:
                        found_pred = cand_pred.id
                        break
                eff_pred_id = found_pred
                if eff_pred_id is None and not is_evading:
                    # All upstream vehicles are evading: this vehicle temporarily acts as lane anchor
                    gap_override = 15.0

            states[veh.id] = PlatoonVehicleCoordinationState(
                vehicle_id=veh.id,
                role=veh.role,
                is_evading=is_evading,
                evasion_state=ev_state,
                lateral_offset_m=lat_off,
                effective_predecessor_id=eff_pred_id,
                desired_gap_override_m=gap_override,
            )

        # 2. Strict Follower Non-Copying Rule:
        # A vehicle that is NOT independently in an active evasion state must NEVER execute
        # an evasive lane change (CHANGE_LEFT / CHANGE_RIGHT) just because a predecessor did!
        for idx in range(num_veh):
            veh = snapshot.platoon[idx]
            act = coordinated[idx]
            is_evading = veh.id in active_evaders

            if not is_evading and act in ("CHANGE_LEFT", "CHANGE_RIGHT"):
                # Mask out unsolicited lane changes from non-threatened followers
                coordinated[idx] = "CRUISE"

        self.coordination_states = states
        return coordinated, states

