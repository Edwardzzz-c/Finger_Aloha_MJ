import re
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from optimizer_helpers import get_sim_prop, simulate_with_new_properties
from dm_control import mjcf


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

# ---------------------------------------------------------------------
# --------------------  LOAD MODEL, DATA, CSV  ------------------------
# ---------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_XML)
data  = mujoco.MjData(model)
  
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
row = 0
sim_t = 0.0
F_cmd = {}
T_cmd = {}

def simulate_error(model, data):

    pos_err_sum, rot_err_sum = 0.0, 0.0
    row, sim_t = 0, 0.0

    while row < len(df):
        if sim_t >= time_series[row]:
            row += 1
        tgt_row = row - 1                      

        for tracker, body_id in BIDS.items():
            p_des = poses[tracker]["pos"][tgt_row]
            q_des = poses[tracker]["quat"][tgt_row]  # in xyzw

            if p_des is None or q_des is None:
                continue                       
            set_body_pose(model, data, body_id, p_des, q_des)
            
        for _ in range(5):
            mujoco.mj_step(model, data)
        sim_t = data.time

        for tracker, body_id in BIDS.items():
            p_des = poses[tracker]["pos"][tgt_row]
            q_des = poses[tracker]["quat"][tgt_row]

            if p_des is None or q_des is None:
                continue

            p_cur = data.xpos[body_id]         
            q_cur = data.xquat[body_id]         

            # Reorder q_des to [w, x, y, z]
            q_des_wxyz = np.empty(4)
            q_des_wxyz[0] = q_des[3]
            q_des_wxyz[1:] = q_des[:3]

            # Position error
            pos_err = np.linalg.norm(p_cur - p_des)

            # Rotation error: angle between quaternions
            R_des = R.from_quat(q_des_wxyz)
            R_cur = R.from_quat(q_cur)
            R_rel = R_des * R_cur.inv()
            rot_err = R_rel.magnitude()  # radians

            pos_err_sum += pos_err
            rot_err_sum += rot_err

    return pos_err_sum + rot_err_sum

# incomplete
def main_loop(start, end):
    best_propdict = None
    lowest_err = float('inf')
    property_dict = create_prop_dict("proxi_exo1", "0 0 0 -.012 0 .005")
    model, data = simulate_with_new_properties(MODEL_XML, property_dict)
    error = simulate_error(model, data)
    print(error)

BODIES = ["proxi_exo1", "proxi_exo2", "distal_exo1", "distal_exo2", "distal_exo3"]
PARAMS = ["fromto", "size"]

def create_prop_dict(part, val):

    mjcf_tree = mjcf.from_path(MODEL_XML)
    val_list = val.split()
    from_ = ' '.join(val_list[:3])
    to_ = ' '.join(val_list[3:])

    #import ipdb;ipdb.set_trace()
    if part == BODIES[0]:
        proxi_2_val = get_sim_prop(
            mjcf_tree, BODIES[1], "geom", PARAMS[0])
        print(proxi_2_val)
        lst = proxi_2_val.split()
        proxi_2_new = to_ + ' ' + ' '.join(lst[3:])
        property_dict = {(BODIES[0], "geom", PARAMS[0]): val,
                         (BODIES[1], "geom", PARAMS[0]): proxi_2_new}
        
    elif part == BODIES[1]:
        proxi_1_val = get_sim_prop(
            mjcf_tree, BODIES[0], "geom", PARAMS[0])
        lst = proxi_1_val.split()
        proxi_1_new = ' '.join(lst[:3]) + ' ' + from_
        property_dict = {(BODIES[0], "geom", PARAMS[0]): proxi_1_new,
                         (BODIES[1], "geom", PARAMS[0]): val}
        
    elif part == BODIES[2]:
        distal_2_val = get_sim_prop(
            mjcf_tree, BODIES[3], "geom", PARAMS[0])
        lst = distal_2_val.split()
        distal_2_new = to_ + ' ' + ' '.join(lst[3:])
        property_dict = {(BODIES[2], "geom", PARAMS[0]): val,
                         (BODIES[3], "geom", PARAMS[0]): distal_2_new,}
        
    elif part == BODIES[3]:
        distal_1_val = get_sim_prop(
            mjcf_tree, BODIES[2], "geom", PARAMS[0])
        lst = distal_1_val.split()
        distal_1_new = ' '.join(lst[:3]) + ' ' + from_
        distal_3_val = get_sim_prop(
            mjcf_tree, BODIES[4], "geom", PARAMS[0])
        lst = distal_3_val.split()
        distal_3_new = to_ + ' ' + ' '.join(lst[3:]) 
        property_dict = {(BODIES[2], "geom", PARAMS[0]): distal_1_new,
                         (BODIES[3], "geom", PARAMS[0]): val,
                         (BODIES[4], "geom", PARAMS[0]): distal_3_new}
        
    elif part == BODIES[4]:
        distal_2_val = get_sim_prop(
            mjcf_tree, BODIES[3], "geom", PARAMS[0])
        lst = distal_2_val.split()
        distal_2_new = ' '.join(lst[:3]) + ' ' + from_
        property_dict = {(BODIES[2], "geom", PARAMS[0]): distal_2_new,
                         (BODIES[3], "geom", PARAMS[0]): val}
        
    return property_dict

main_loop(1,1)