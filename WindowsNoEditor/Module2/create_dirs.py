import os

base_dir = r"d:\I FILES\Studies\sem7\MiniProj\WindowsNoEditor\Module2"

dirs = [
    "observation",
    "mappo",
    "action",
    "reward",
    "integration",
    "checkpoints"
]

for d in dirs:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    init_file = os.path.join(base_dir, d, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            pass

# Also create base __init__.py
base_init = os.path.join(base_dir, "__init__.py")
if not os.path.exists(base_init):
    with open(base_init, "w") as f:
        pass

print("Directories created.")

