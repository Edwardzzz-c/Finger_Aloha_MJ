import mujoco
from dm_control import mjcf
import mujoco.viewer
from optimizer import Optimizer
"""
# 1) Load XML into dm_control tree
root = mjcf.from_path("square.xml")

# Edits to dm_control model
body = root.find("body","hi")
geom = body.find("geom", "box")
geom.rgba = "1 0 0 1"

# 2) Turn MJCF tree back to xml_string and feed in to Mujoco parser
model = mujoco.MjModel.from_xml_string(root.to_xml_string())
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)
"""


# 4) View

opt = Optimizer()
root = mjcf.from_path("square.xml")
opt.set_sim_body_property(root, "hi", "geom", "rgba", "1 1 0 1")
model, data = opt.get_sim_from_xml(root)
viewer = mujoco.viewer.launch_passive(model, data)
while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
