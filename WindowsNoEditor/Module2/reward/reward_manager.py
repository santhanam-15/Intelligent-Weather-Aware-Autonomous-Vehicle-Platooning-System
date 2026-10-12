import math

import carla
import numpy as np

from .. import config


class RewardManager:
    def __init__(self, uncertainty_manager):
        self.uncertainty_manager = uncertainty_manager
        self.control_context = None
        self.last_components = []

    def set_control_context(self, controller):
        self.control_context = controller

    def _context(self, vehicle, index):
        try:
            return self.control_context.observation_context(vehicle, index) if self.control_context else {}
        except Exception:
            return {}

    @staticmethod
    def _weather_name(weather):
        current = getattr(weather, "current_preset", None)
        if current is None:
            current = getattr(weather, "current_weather", "CLEAR")
        return getattr(current, "name", str(current)).upper().split(".")[-1]

    def _get_friction(self, vehicle=None):
        try:
            friction_manager = getattr(self.uncertainty_manager, "friction", None)
            if friction_manager is not None and vehicle is not None:
                managed_mu = friction_manager.get_friction(vehicle)
                if managed_mu is not None:
                    return float(managed_mu)
        except Exception:
            pass

        try:
            weather = getattr(self.uncertainty_manager, "weather", None)
            return {
                "CLEAR": 1.0,
                "CLOUDY": 0.9,
                "LIGHT_RAIN": 0.6,
                "MODERATE_RAIN": 0.6,
                "HEAVY_RAIN": 0.4,
                "FOG": 0.7,
                "HEAVY_FOG": 0.7,
                "RAIN_FOG": 0.4,
            }.get(self._weather_name(weather), 1.0)
        except Exception:
            return 1.0

    def calculate_desired_headway(self, speed, rel_speed, friction):
        """Weather-aware desired clearance in metres."""
        del rel_speed  # retained in the interface for future car-following terms
        base = config.DEFAULT_DESIRED_HEADWAY
        speed_factor = max(0.0, speed) * 0.5
        friction_factor = (1.0 - friction) * 10.0
        return base + speed_factor + friction_factor

    def compute_rewards(self, platoon, actions=None):
        """Compute one reward for each ordered platoon vehicle.

        Headway is intentionally measured against the immediate platoon
        predecessor, not an arbitrary NPC selected from the CARLA world.
        ``actions`` is accepted for API compatibility; the reward terms are
        state-based and therefore remain associated with each vehicle's state.
        """
        del actions
        rewards = []
        self.last_components = []

        for i, vehicle in enumerate(platoon):
            if not vehicle or not vehicle.is_alive:
                rewards.append(-10.0)
                continue

            reward = 0.0
            components = {"traffic": 0.0, "headway": 0.0, "efficiency": 0.0, "route": 0.0, "safety": 0.0}
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

            tl_state = vehicle.get_traffic_light_state()
            if vehicle.is_at_traffic_light() and tl_state == carla.TrafficLightState.Red:
                if speed > 0.5:
                    reward += config.RED_LIGHT_PENALTY
                    components["traffic"] += config.RED_LIGHT_PENALTY

            predecessor_map = getattr(self.control_context, "predecessor_map", None) if self.control_context else None
            pred_idx = predecessor_map.get(i) if predecessor_map is not None else (i - 1 if i > 0 else None)
            disabled_vehicles = getattr(self.control_context, "disabled_vehicles", set()) if self.control_context else set()

            if pred_idx is not None and 0 <= pred_idx < len(platoon) and vehicle.id not in disabled_vehicles:
                ahead = platoon[pred_idx]
                if ahead and ahead.is_alive and ahead.id not in disabled_vehicles:
                    center_distance = vehicle.get_location().distance(ahead.get_location())
                    clearance = center_distance - 4.7
                    if clearance <= 0.0:
                        reward += config.COLLISION_PENALTY
                        components["safety"] += config.COLLISION_PENALTY
                    else:
                        ahead_velocity = ahead.get_velocity()
                        ahead_speed = math.sqrt(
                            ahead_velocity.x ** 2
                            + ahead_velocity.y ** 2
                            + ahead_velocity.z ** 2
                        )
                        rel_speed = ahead_speed - speed
                        desired_hw = self.calculate_desired_headway(
                            speed, rel_speed, self._get_friction(vehicle)
                        )

                        if clearance < desired_hw * 0.5:
                            reward += config.UNSAFE_HEADWAY_PENALTY
                            components["headway"] += config.UNSAFE_HEADWAY_PENALTY
                        elif abs(clearance - desired_hw) <= config.HEADWAY_DEADBAND:
                            reward += config.GOOD_HEADWAY_REWARD
                            components["headway"] += config.GOOD_HEADWAY_REWARD
                        elif clearance > desired_hw * 1.5:
                            reward -= 0.5
                            components["headway"] -= 0.5

            if speed > 5.0 and tl_state != carla.TrafficLightState.Red:
                reward += config.EFFICIENCY_REWARD
                components["efficiency"] += config.EFFICIENCY_REWARD

            context = self._context(vehicle, i)
            # A route-progress term is bounded and only rewards actual
            # completion progress, not arbitrary motion or lane wandering.
            route_reward = min(0.2, max(0.0, float(context.get("route_progress", 0.0)) * 0.2))
            reward += route_reward
            components["route"] += route_reward
            if context.get("safety_override"):
                components["safety"] -= 0.02
                reward -= 0.02

            rewards.append(float(reward))
            self.last_components.append(components)

        collective_reward = sum(rewards) / max(1, len(rewards))
        final_rewards = [
            float(reward + config.COLLECTIVE_REWARD_WEIGHT * collective_reward)
            for reward in rewards
        ]
        return np.nan_to_num(
            np.asarray(final_rewards, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
