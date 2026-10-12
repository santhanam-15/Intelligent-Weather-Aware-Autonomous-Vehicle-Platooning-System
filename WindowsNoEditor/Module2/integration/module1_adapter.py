class Module1Adapter:
    """Translate one tactical action into one vehicle-specific controller input."""

    def apply_actions(self, tactical_actions, leader_controller, platoon_controller, platoon):
        if not leader_controller or not platoon_controller:
            return

        for index, vehicle in enumerate(platoon):
            if not vehicle or index >= len(tactical_actions):
                continue

            action = tactical_actions[index]
            if index == 0:
                if hasattr(leader_controller, "apply_tactical_action"):
                    leader_controller.apply_tactical_action(action)
                else:
                    if action == "ACCELERATE":
                        leader_controller.target_speed_kmh = min(
                            60.0, leader_controller.target_speed_kmh + 1.0
                        )
                    elif action == "DECELERATE":
                        leader_controller.target_speed_kmh = max(
                            0.0, leader_controller.target_speed_kmh - 1.0
                        )
                    elif action == "CHANGE_LEFT":
                        leader_controller.request_lane_change("LEFT")
                    elif action == "CHANGE_RIGHT":
                        leader_controller.request_lane_change("RIGHT")
            else:
                # The controller stores a separate clearance/action state per
                # follower.  No follower action is aggregated into another
                # follower's command.
                platoon_controller.apply_tactical_action(vehicle, action)
