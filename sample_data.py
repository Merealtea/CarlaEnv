from env.BEV_env import Env
import yaml
from datetime import datetime
import carla

with open("./config/env.yaml") as f:
    hype = yaml.safe_load(f)

current_time = datetime.now()
env_name = current_time.strftime("%Y_%m_%d_%H_%M_%S")

sim_settings = hype["simulation_settings"]

sp = carla.Location(x=-47.711586, y=-41.689964, z=0.450000)

nv_trans = [
    carla.Transform(
        location = carla.Location(x = sp.x + 25, y = sp.y + 23, z = sp.z),
        rotation = carla.Rotation(yaw = 180)
    ),
    carla.Transform(
        location = carla.Location(x = sp.x + 25, y = sp.y + 20, z = sp.z),
        rotation = carla.Rotation(yaw = 180)
    ),

    carla.Transform(
        location = carla.Location(x = sp.x + 25, y = sp.y + 17, z = sp.z),
        rotation = carla.Rotation(yaw = 180)
    ),

    carla.Transform(
        location = carla.Location(x = sp.x + 30, y = sp.y + 15, z = sp.z),
        rotation = carla.Rotation(yaw = 90)
    )
]
             

cav_trans = [ 
    carla.Transform(
        location = carla.Location(x = sp.x + 20, y = sp.y + 28, z = sp.z),
        rotation = carla.Rotation(yaw = 0)
    ),

    carla.Transform(
        location = carla.Location(x = sp.x + 34.5, y = sp.y + 20, z = sp.z),
        rotation = carla.Rotation(yaw = 0)
    ) 
]
predesigned_trans = [

]
carla_env = Env(env_name = env_name, cav_trans = cav_trans,
                   neighbor_trans = nv_trans,
                   use_autopilot = [True] + [False] * (len(cav_trans)-1) + [False] * len(nv_trans),
                   **hype['env'])

# try:
carla_env.simulate(sim_settings["simulation_steps"])
# except:
#     carla_env.destroy_world()
# finally:

# carla_env.validation_check()
carla_env.destroy_world()

