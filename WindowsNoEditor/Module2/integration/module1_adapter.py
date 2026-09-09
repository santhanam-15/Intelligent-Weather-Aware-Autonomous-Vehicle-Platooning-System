from .. import config

class Module1Adapter:
    def apply_actions(self, tactical_actions, leader_controller, platoon_controller):
        """
        tactical_actions: list of strings e.g. ["CRUISE", "DECELERATE", ...]
        Applies changes to the low-level controllers from Module 1.
        """
        if not leader_controller or not platoon_controller:
            return

        for i, action in enumerate(tactical_actions):
            if i == 0:
                # Leader Controller
                # It uses target_speed_kmh
                if action == "ACCELERATE":
                    leader_controller.target_speed_kmh = min(60.0, leader_controller.target_speed_kmh + 2.0)
                elif action == "DECELERATE":
                    leader_controller.target_speed_kmh = max(0.0, leader_controller.target_speed_kmh - 5.0)
                elif action == "CRUISE":
                    pass
            else:
                # Follower Controller
                # It uses desired_clearance
                # A simplistic mapping: DECELERATE increases desired clearance, ACCELERATE decreases it.
                if action == "DECELERATE":
                    platoon_controller.desired_clearance = min(25.0, platoon_controller.desired_clearance + 2.0)
                elif action == "ACCELERATE":
                    platoon_controller.desired_clearance = max(5.0, platoon_controller.desired_clearance - 1.0)
                elif action == "CRUISE":
                    pass

        # To prevent extreme wobbling in the low-level controller
        platoon_controller.desired_clearance = max(5.0, min(25.0, platoon_controller.desired_clearance))

