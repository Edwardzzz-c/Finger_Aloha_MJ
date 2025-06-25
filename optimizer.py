import re
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from optimizer_helpers import simulate_with_new_properties
from dm_control import mjcf
from tqdm import tqdm

# ---------------------------------------------------------------------
# -------------------------  USER SETTINGS  ---------------------------
# ---------------------------------------------------------------------
MODEL_XML = "index_finger.xml"      
CSV_FILE  = "finger_kinematics_data/Jun20.3.csv" 
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
R_corr = R.from_euler("xy", [-15,-13], degrees=True)
SCALE  = 1.0            
OFFSET = np.array([0.2,0,-0.035])

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


def create_prop_dict(part, val):
    """
    Create a property dictionary for the specified part and value.
    The value is expected to be a 'from to' list."""

    from_ = val[:3]
    to_ = val[3:]
    
    if part == "proxi_exo1":
        proxi_2_new_pos = to_
        property_dict = {("proxi_exo1", "geom", "fromto"): val,
                         ("proxi_exo2", None, "pos"): proxi_2_new_pos}
      
    elif part == "proxi_exo2":
        property_dict = {("proxi_exo2", "geom", "fromto"): val}
        
    elif part == "distal_exo1":
        
        distal_2_new_pos = to_
        property_dict = {("distal_exo1", "geom", "fromto"): val,
                         ("distal_exo2", None, "pos"): distal_2_new_pos}
 
    elif part == "distal_exo2":
        property_dict = {("distal_exo2", "geom", "fromto"): val,
                         ("distal_exo3", None, "pos"): to_}
        
    elif part == "distal_exo3":
        property_dict = {("distal_exo3", "geom", "fromto"): val}
        
    return property_dict


def cem_sampling(link, L_min, L_max,
                 n_epochs=40, n_samples=20, best_frac=0.2, seed=0, k=3,):
    
    """Cross-Entropy search that returns the best length found.
    Args:
        n_epochs (int): Number of epochs to run.
        n_samples (int): Number of samples per epoch.
        best_frac (float): Fraction of best samples to keep.
        seed (int): Random seed for reproducibility.
        L_min (float): Minimum length to sample.
        L_max (float): Maximum length to sample.
        k (int): Scaling factor for standard deviation.
        link (str): The body name to optimize.
    Returns:
        tuple: Best length found and its corresponding score."""
    
    root = mjcf.from_path(MODEL_XML)

    rng = np.random.default_rng(seed)
    mean = (L_min + L_max) / 2
    std  = (L_max - L_min) / (2 * k)
    n_best = max(1, int(n_samples * best_frac))
    best_param, best_len, best_score = None, None, -np.inf

    for _ in tqdm(range(n_epochs)):
        lens   = rng.standard_normal(n_samples) * std + mean
        scores = np.empty(n_samples)

        for i, l in enumerate(lens):
            from_to = len_to_fromto(l, link)
            property_dict = create_prop_dict(link, from_to)
            model, data = simulate_with_new_properties(root, property_dict)
            s = -simulate_error(model, data)
            scores[i] = s
            if s > best_score:                     
                best_len, best_score = l, s
                best_param = from_to

        elite = lens[scores.argsort()[-n_best:]]    
        mean, std = elite.mean(), elite.std() + 1e-6

    return best_param, best_len, -best_score
    
def len_to_fromto(length, link):
    """
    Converts a length to a 'fromto' string format.
    """
    if link == "distal_exo2":
        ux, uy, uz = -0.148523, 0.0, 0.988909            # unit vector
        x, y, z = ux*length, uy*length, uz*length
        return [0,0,0,round(x,6),round(y,6),round(z,6)]

if __name__ == "__main__":
    best_param, best_len, best_score = cem_sampling(link="distal_exo2", L_min=0.05, L_max=0.13)
    print("Optimization complete.")
    best_param = [float(v) for v in best_param]
    print(f"Best parameter: {best_param}", 
          f"Best length: {best_len:.6f}",
          f"Best score: {best_score:.6f}")