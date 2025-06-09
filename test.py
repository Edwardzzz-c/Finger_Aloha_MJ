import mujoco
import numpy as np

# Load the model
MODEL_PATH = "index_finger.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Control variables for the finger joints
ctrl_prox = 0
ctrl_dist = 0
LOW = 0.0
HIGH = 1.57  # max range (radians) of movement
STEP = 0.05  # angle (radians) change per key press

# Key callback to control finger joints
def key_callback(key):
    global ctrl_prox, ctrl_dist
    if key == ord('O'):           # bend proximal
        ctrl_prox = min(ctrl_prox + STEP, HIGH)
    elif key == ord('P'):         # straighten proximal
        ctrl_prox = max(ctrl_prox - STEP, LOW)
    elif key == ord('K'):         # bend distal
        ctrl_dist = min(ctrl_dist + STEP, HIGH)
    elif key == ord('L'):         # straighten distal
        ctrl_dist = max(ctrl_dist - STEP, LOW)
    elif key == ord('R'):         # reset all joints
        ctrl_prox = 0.0
        ctrl_dist = 0.0
    print(f"Proximal: {ctrl_prox:.2f}, Distal: {ctrl_dist:.2f}")

# Launch the viewer in passive mode
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    # Camera settings
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.2

    # Main simulation loop
    while viewer.is_running():
        # Apply control inputs for the finger joints
        data.ctrl[0] = ctrl_prox
        data.ctrl[1] = ctrl_dist

        # Step the simulation
        mujoco.mj_step(model, data)

        # Sync the viewer (this also applies mouse perturbations for dragging)
        viewer.sync()