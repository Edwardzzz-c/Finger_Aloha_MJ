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
CSV_FILE  = "finger_kinematics_data/new_data2.csv" 
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
OFFSET = np.array([-0.03, -0.22, 0]) """
# Edward's transform
R_corr = R.from_euler("y", 180, degrees=True)
SCALE  = 1.0            
OFFSET = np.array([-0.195, 0.083, -0.04])  

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

# ------------------------------------------------------------------
# Helper: overwrite a free body’s world-pose
# ------------------------------------------------------------------
def set_body_pose(model, data, body_id, p_xyz, q_xyzw):
    """
    p_xyz  : (3,) world position  [x y z]
    q_xyzw : (4,) world quaternion (x y z w)

    """
    joint_adr   = model.body_jntadr[body_id]      
    qpos_adr   = model.jnt_qposadr[joint_adr]         

    # Reorder to wxyz
    qwxyz = np.empty(4)
    qwxyz[0] = q_xyzw[3]
    qwxyz[1:] = q_xyzw[:3]

    data.qpos[qpos_adr     : qpos_adr+3] = p_xyz      # x, y, z
    data.qpos[qpos_adr+3   : qpos_adr+7] = qwxyz      # w, x, y, z

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
    for tracker, body_id in BIDS.items():
        # Get the target pose from poses dict
        p_des = poses[tracker]["pos"][tgt_row]
        q_des = poses[tracker]["quat"][tgt_row]  # in xyzw

        if p_des is None or q_des is None:
            continue                       
        set_body_pose(model, data, body_id, p_des, q_des)
        
    for _ in range(5):
        mujoco.mj_step(model, data)
    sim_t = data.time
    
    viewer.sync()
    time.sleep(0.01)
    """if row % 100 == 0:
        for tracker, body_id in BIDS.items():
            p_des = poses[tracker]['pos'][tgt_row]
            p_cur = data.xpos[body_id]
            print(f"{tracker}: p_des = {p_des}, p_cur = {p_cur}")"""

            

viewer.close()
