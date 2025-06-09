import mujoco
import numpy as np
import pandas as pd


MODEL_PATH = "index_finger.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# ── sensor address for tip force (3-vector) ───────────────────────────────────
SENSOR_ID = model.sensor(name="tip_force").id


def print_tip_force():
    fx, fy, fz = data.sensordata[SENSOR_ID:SENSOR_ID+3]
    print(f"tip force [N] {fx: .3f} {fy: .3f} {fz: .3f}")

with mujoco.viewer.launch_passive(model, data,) as viewer:
    viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 90, -20, 0.25

    while viewer.is_running():
        # write every desired angle into its actuator
       # for name, idx in ACT.items():
        #    data.ctrl[idx] = ctrl[name]

        mujoco.mj_step(model, data)
       # print_tip_force()
        viewer.sync()
