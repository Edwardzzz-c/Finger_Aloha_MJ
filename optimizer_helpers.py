import mujoco
from dm_control import mjcf
import os


def get_sim_from_mjcf(root: mjcf.RootElement):       

    mjcf.export_with_assets(root, "tmpdir", "model.xml")
    xml_path = os.path.join("tmpdir", "model.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    return model, data
  

def set_sim_prop(mjcf_tree: mjcf.RootElement,
                          body_name: str,
                          element_name: str,
                          prop_name: str,
                          value: list):
  """
  Helper function to set specific body property
  Usage example: set_sim_body_property(root, "proxi_exo1", "geom", "rgba", [0,0,0,1,1,1])
  """
  
  body = mjcf_tree.find('body', body_name)
  if body is None:
      raise ValueError(f'No <body name="{body_name}"> found in MJCF')
  if element_name is None:
        if not hasattr(body, prop_name):
            raise ValueError(
                f'Body "{body_name}" has no attribute "{prop_name}"'
            )
        setattr(body, prop_name, value)
        return
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

def get_sim_prop(mjcf_tree: mjcf.RootElement, 
                 body_name: str,
                 element_name: str, 
                 prop_name: str):
    """
    Helper function to get specific body property
    Usage example: get_sim_prop(root, "proxi_exo1", "geom", "rgba")
    return numpy array of the property value
    If element_name is None, returns the property of the body itself.
    """
    body = mjcf_tree.find('body', body_name)
    if body is None:
        raise ValueError(f'No <body name="{body_name}"> found in MJCF')
    if element_name is None:
        return getattr(body, prop_name)
        
    
    elems = body.find_all(element_name)
    if not elems:
        raise ValueError(f'Body "{body_name}" has no <{element_name}> children')
        # Gets value from first element of this type in body!
    elem = elems[0]
    if not hasattr(elem, prop_name):
        raise ValueError(
            f'<{element_name}> in body "{body_name}" has no attribute "{prop_name}"'
        )
    
    return getattr(elem, prop_name)

def simulate_with_new_properties(root, property_dict):
    """
        @param property_dict = {(bodyname, element_name, property name): new value}
    """

    for (body_name, elem_name, prop_name), value in property_dict.items():
        set_sim_prop(root, body_name, elem_name, prop_name, value)

    try:
        model, data = get_sim_from_mjcf(root)
    except ValueError as err:
        raise RuntimeError(f'MuJoCo parser failed after edits: {err}') from err

    return model, data
