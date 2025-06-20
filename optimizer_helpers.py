import mujoco
from dm_control import mjcf
import os

def get_sim_from_mjcf(root: mjcf.RootElement):

  model = mujoco.MjModel.from_xml_string(root.to_xml_string())
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
 
  return model, data
  

def set_sim_prop(mjcf_tree: mjcf.RootElement,
                          body_name: str,
                          element_name: str,
                          prop_name: str,
                          value: str):
  """
  Helper function to set specific body property
  Usage example: set_sim_body_property(root, "proxi_exo1", "geom", "rgba", "0 0 1 1")
  """
  
  body = mjcf_tree.find('body', body_name)
  if body is None:
      raise ValueError(f'No <body name="{body_name}"> found in MJCF')

  elems = body.find_all(element_name)

  if not elems:
      raise ValueError(f'Body "{body_name}" has no <{element_name}> children')

  # Sets value to all element of this type in body!
  for elem in elems:
      if not hasattr(elem, prop_name):
          raise ValueError(
              f'<{element_name}> in body "{body_name}" has no attribute "{prop_name}"'
          )
      setattr(elem, prop_name, value)

def get_sim_prop(mjcf_tree: mjcf.RootElement, body_name: str, element_name: str, prop_name: str):
    """
    Helper function to get specific body property
    Usage example: get_sim_prop(root, "proxi_exo1", "geom", "rgba")
    """
    body = mjcf_tree.find('body', body_name)
    if body is None:
        raise ValueError(f'No <body name="{body_name}"> found in MJCF')
    elems = body.find_all(element_name)
    if not elems:
        raise ValueError(f'Body "{body_name}" has no <{element_name}> children')
    
    body = mjcf_tree.find('body', body_name)
   
    # Gets value from first element of this type in body!
    elem = elems[0]
    if not hasattr(elem, prop_name):
        raise ValueError(
            f'<{element_name}> in body "{body_name}" has no attribute "{prop_name}"'
        )
    
    arr = getattr(elem, prop_name)
    return ' '.join(str(x) for x in arr)

def simulate_with_new_properties(xml, property_dict):
    """
        @param property_dict = {(bodyname, element_name, property name): new value}
    """
    root = mjcf.from_path(xml)

    for (body_name, elem_name, prop_name), value in property_dict.items():
        set_sim_prop(root, body_name, elem_name, prop_name, value)

    try:
        model, data = get_sim_from_mjcf(root)
    except ValueError as err:
        raise RuntimeError(f'MuJoCo parser failed after edits: {err}') from err

    return model, data
