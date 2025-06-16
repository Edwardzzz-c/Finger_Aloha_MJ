import re
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import time
import optimizer

# ---------------------------------------------------------------------
# -------------------------  USER SETTINGS  ---------------------------
# ---------------------------------------------------------------------
MODEL_XML = "index_finger.xml"      
CSV_FILE  = "finger_kinematics_data/finger_calibration_2.csv" 
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
R_corr = R.from_euler("z", -90, degrees=True)
SCALE  = 1.0            
OFFSET = np.array([-0.03, -0.22, 0])   

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
row = 0
sim_t = 0.0
F_cmd = {}
T_cmd = {}

def simulate_error(model, data, KP_POS=8e-3, KP_ROT=8e-3):
    """Run the replay once (50 internal sub-steps per video frame) and
       return total position-plus-orientation tracking error."""

    pos_err_sum, rot_err_sum = 0.0, 0.0
    row, sim_t = 0, 0.0

    while row < len(df):
        if sim_t >= time_series[row]:
            row += 1
        tgt = row - 1

        # --------- compute forces / torques for each body ---------
        data.xfrc_applied[:] = 0.0
        for col, bid in BIDS.items():
            p_des = poses[col]["pos"][tgt]
            q_des = poses[col]["quat"][tgt]
            p_cur = data.xpos[bid].copy()
            q_cur = data.xquat[bid][[1, 2, 3, 0]]  # xyzw

            # position error and force
            dp = np.zeros(3) if p_des is None else (p_des - p_cur)
            F  = KP_POS * dp

            # rotation error and torque
            if q_des is None:
                rotvec = np.zeros(3)
                T_world = np.zeros(3)
            else:
                rotvec   = quat_err(q_des, q_cur)
                T_body   = KP_ROT * rotvec
                R_world  = R.from_quat(q_cur).as_matrix()
                T_world  = R_world @ T_body

            data.xfrc_applied[bid, :3] = F
            data.xfrc_applied[bid, 3:] = T_world

            pos_err_sum += dp @ dp
            rot_err_sum += rotvec @ rotvec

        # --------- advance physics *exactly* 50 times -----------
        for _ in range(50):          
            mujoco.mj_step(model, data)
            sim_t += model.opt.timestep

    return pos_err_sum + rot_err_sum

def main_loop(start, end):
    best_propdict = None
    lowest_err = float('inf')
    for val in range(start, end):
        property_dict = create_prop_dict(len)
        optim = optimizer.Optimizer()
        model, data = optim.simulate_with_new_properties(MODEL_XML, property_dict)
        error = simulate_error(model, data)
        if error < lowest_err:
            lowest_err = error
            best_propdict = property_dict
    return best_propdict

BODIES = ["proxi_exo1", "proxi_exo2", "distal_exo1", "distal_exo2", "distal_exo3"]
PARAMS = ["fromto", "size", "pos"]

def create_prop_dict(val, part):

    if part == BODIES[2]:
        property_dict = {(part, "geom", PARAMS[0]): val}

    return property_dict

