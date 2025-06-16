import re
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import time

# ---------------------------------------------------------------------
# -------------------------  USER SETTINGS  ---------------------------
# ---------------------------------------------------------------------
MODEL_XML = "index_finger.xml"      
CSV_FILE  = "finger_kinematics_data/new_.csv" 
BODY_MAP  = {                        
    "trakstar0": "shell_dist",
    "trakstar1": "shell_mid",
    "trakstar2": "shell_prox",
}


# ---------------------------------------------------------------------
# -----------------------  UTILITIES & HELPERS  -----------------------
# ---------------------------------------------------------------------
# Compile regex pattern
_POSE_RE_TR = re.compile(
    r"translation:.*?x:\s*([-\d.eE]+).*?"
    r"y:\s*([-\d.eE]+).*?z:\s*([-\d.eE]+)",
    re.S | re.I,
)
_POSE_RE_RT = re.compile(
    r"rotation:.*?x:\s*([-\d.eE]+).*?"
    r"y:\s*([-\d.eE]+).*?z:\s*([-\d.eE]+).*?"
    r"w:\s*([-\d.eE]+)",
    re.S | re.I,
)

def _parse_pose_block(cell: str):
    """
    Extract (pos[3], quat[4]) from one trakstar column string.
    Returns None if it can’t find both translation & rotation blocks.
    """
    if not isinstance(cell, str) or "translation" not in cell:
        return None

    mt = _POSE_RE_TR.search(cell)
    mr = _POSE_RE_RT.search(cell)
    if not (mt and mr):
        return None                  

    # translation x,y,z
    tx, ty, tz = map(float, mt.groups())

    # rotation x,y,z,w
    rx, ry, rz, rw = map(float, mr.groups())
    quat = np.array([rx, ry, rz, rw], dtype=float)
    quat /= np.linalg.norm(quat)          # normalise 

    pos = np.array([tx, ty, tz], dtype=float)
    return pos, quat


def quat_err(qd, qc):
    """axis-angle error (body frame) that rotates qc → qd."""
    return (R.from_quat(qc).inv() * R.from_quat(qd)).as_rotvec()


# ---------------------------------------------------------------------
# --------------------  LOAD MODEL, DATA, CSV  ------------------------
# ---------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_XML)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)   
# Map body IDs once for speed
BIDS = {col: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        for col, body in BODY_MAP.items()}

df = pd.read_csv(CSV_FILE)

# Pre-extract time column 
time_series = df["time_elapsed"].values.astype(float)

# For each tracker column we care about, parse every row up-front
poses = {}
for col in BODY_MAP:
    pos_list, quat_list = [], []
    for cell in df[col]:
        parsed = _parse_pose_block(cell)
        if parsed is None:
            pos_list.append(None)
            quat_list.append(None)
        else:
            pos_list.append(parsed[0])
            quat_list.append(parsed[1])
    poses[col] = dict(pos=pos_list, quat=quat_list)

# ---------------------------------------------------------------------
# ---------------------  Transform & Translation ----------------------
# ---------------------------------------------------------------------
# Katelyn's transform
"""R_corr = R.from_euler("z", -90, degrees=True)
SCALE  = 1.0            
OFFSET = np.array([-0.03, -0.22, 0])""" 
# Edward's transform
R_corr = R.from_euler("y", 180, degrees=True)
SCALE  = 1.0            
OFFSET = np.array([-0.195, 0.078, -0.015])  

for col in poses:
    # transform every position
    poses[col]["pos"] = [
        None if p is None else R_corr.apply(p * SCALE) + OFFSET
        for p in poses[col]["pos"]
    ]
    # transform every quaternion
    poses[col]["quat"] = [
        None if q is None else (R_corr * R.from_quat(q)).as_quat()
        for q in poses[col]["quat"]
    ]

# ---------------------------------------------------------------------
# ----------------------------  SIM LOOP  -----------------------------
# ---------------------------------------------------------------------
viewer = mujoco.viewer.launch_passive(model, data) 
row = 0
sim_t = 0.0
F_cmd = {}
T_cmd = {}

while viewer.is_running() and row < len(df):
    # Advance row pointer whenever we've passed that timestamp
    if sim_t >= time_series[row]:
        row += 1
    tgt_row = row - 1                      

    # Loop over each controlled body
    for col, body_id in BIDS.items():
        # Get the target pose from poses dict
        p_des = poses[col]["pos"][tgt_row]
        q_des = poses[col]["quat"][tgt_row]  # in xyzw

        # Current world position/world orientation
        p_cur = data.xpos[body_id].copy()
        q_cur = data.xquat[body_id].copy()   # in wxyz
        q_cur = q_cur[[1,2,3,0]]  # now in xyzw
        
        # Current linear/angular velocity
        v_cur = data.cvel[body_id, :3]
        w_cur = data.cvel[body_id, 3:]

        # Pd controller values
        MAX_TORQUE = 8e-2
        MAX_FORCE = 8e-2
        #DAMPENING = 0.0005
        
        # Calculate force/torque to apply to bodies
        if p_des is None:
            F = np.zeros(3)
        else: 
            F = MAX_FORCE * (p_des - p_cur) #- DAMPENING * v_cur
        if q_des is None:
            T_world = np.zeros(3)
        else: 
            rot_err = quat_err(q_des, q_cur)
            T_body  = MAX_TORQUE * rot_err #- DAMPENING * w_cur

            # Convert torque to world frame
            R_world = R.from_quat(q_cur).as_matrix()
            T_world = R_world @ T_body

        F_cmd[body_id] = F     
        T_cmd[body_id] = T_world

        if row % 100 == 0:
            print(f"""pos_desired = {p_des}""")
            print(f"""pos_current = {p_cur}""")
            print(f"""Force applied = {F}""")
        #    print(f"""velocity = {v_cur}""")
            print("rotvec_desired =", R.from_quat(q_des).as_rotvec())
            print("rotvec_current =", R.from_quat(q_cur).as_rotvec())
            print(f"""Rotation applied = {T_world}""")
        
    for _ in range(50):
        for bid in BIDS.values():   
            data.xfrc_applied[bid, :3] = F_cmd[bid]
            data.xfrc_applied[bid, 3:] = T_cmd[bid]
        mujoco.mj_step(model, data)
        sim_t += model.opt.timestep

    viewer.sync()
    time.sleep(0.001)

viewer.close()
