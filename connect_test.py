import carla
import random
import time
import numpy as np
import cv2
import os
import math

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)

# Load world
client.load_world('Town05')
world = client.get_world()

# Set update time interval and synchronous_mode 
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05 #20 fps, 5ms
world.apply_settings(settings)

# Get the blueprints of the world
bp = world.get_blueprint_library()

cyclist_bp = []

# vehicle bps
# TODO: This contains some cyclists, we need to filter it out

vehicle_bp = bp.filter('*vehicle*')
# is_bike = [vehicle_bp.get_attribute('number_of_wheels') == 2]
  
# camera parameters
height = 20
pitch = -90

# RGB Camera
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# semantic camera
semantic_camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')

def transform_transition(src_transform, dst_transform):
    location =carla.Location(
        src_transform.location.x - dst_transform.location.x,
        src_transform.location.y - dst_transform.location.y,
        src_transform.location.z - dst_transform.location.z
    )
    rotation = carla.Rotation(
        src_transform.rotation.pitch - dst_transform.rotation.pitch,
        src_transform.rotation.yaw - dst_transform.rotation.yaw,
        src_transform.rotation.roll - dst_transform.rotation.roll
    )
    return carla.Transform(location, rotation)

def bounding_box(extend, transform):
    x, y, z = extend.x, extend.y, 0
    box = np.array([[-x, y, z], [x, y, z], [x, -y, z], [-x, -y, z]]).T
    center = np.array([[transform.location.x, transform.location.y, transform.location.z]]).T
    yaw = transform.rotation.yaw
    box = box + center# np.array([[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw), 0], [0,0,1]]).dot(box)+ center
    
    box = np.concatenate([box, np.ones((1, box.shape[1]))], axis = 0)
    return box

def intrinsic_map(camera_points, intrinsic):
    pixel_point = camera_points.dot(intrinsic)
    return (pixel_point[:,:2] / pixel_point[:,2:3]).astype(int)


def camera_transform_func(camera_trans, height, width, vehicle_id, frame_id):
    # transform the vehicle bbox into camrea coordination representation
    birdseye_image = np.zeros((height, width, 3), dtype=np.uint8)

    vehicle = world.get_actor(vehicle_id)
    vehicle_trans = vehicle.get_transform()

    # import pdb; pdb.set_trace()
    bbox = bounding_box(vehicle.bounding_box.extent, vehicle_trans)
    # bbox = intrinsic_map(bbox, intrinsic)
    
    # This (4, 4) matrix transforms the points from world to sensor coordinates.
    world_2_camera = np.array(camera_trans.get_inverse_matrix())

    # Transform the points from world space to camera space.
    bbox = np.dot(world_2_camera, bbox)
    # print(bbox)

    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    bbox = np.array([
        bbox[1],
        bbox[2] * -1,
        bbox[0]])

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    bbox = np.dot(intrinsic, bbox)

    # Remember to normalize the x, y values by the 3rd value.
    bbox = np.array([
        bbox[0, :] / bbox[2, :],
        bbox[1, :] / bbox[2, :]]).astype(int).T

    # import pdb; pdb.set_trace()
    # 在鸟瞰图上绘制自车的包围盒
    cv2.polylines(birdseye_image, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)
    if not os.path.exists("./out/bev_gt"):
        os.makedirs("./out/bev_gt/")
    cv2.imwrite("./out/bev_gt/vehicle_{}_{}.png".format(vehicle_id, frame_id), birdseye_image)
    return bbox

# 定义语义分割相机回调函数
def semantic_camera_callback(data, vehicle_id):
    # 获取语义分割图像数据
    semantic_data = np.frombuffer(data.raw_data, dtype=np.uint8)
    semantic_data = np.reshape(semantic_data, (data.height, data.width, 4))[:,:,:3]
    semantic_data = semantic_data[:,:,::-1] * 30
    if not os.path.exists("./out/sematic"):
        os.makedirs("./out/sematic/")
    cv2.imwrite("./out/sematic/vehicle_{}_{}.png".format(vehicle_id, data.frame), semantic_data)

# get the spawn points, every npc will be spawn among these points
spawn_points = world.get_map().get_spawn_points()

sp = spawn_points[124]
num_vehicle = 1
print(sp.location)
ori_loc = carla.Location(x = sp.location.x + 30, y = sp.location.y + 18, z = sp.location.z)
ori_rot = carla.Rotation(yaw = 90)
delta_x = 4
new_trans = [
    # sp,]
             carla.Transform(location = carla.Location(ori_loc.x, ori_loc.y, ori_loc.z), rotation = ori_rot),]
            #  carla.Transform(location = carla.Location(ori_loc.x, ori_loc.y+delta_x, ori_loc.z), rotation = ori_rot),
            #  carla.Transform(location = carla.Location(ori_loc.x-delta_x, ori_loc.y, ori_loc.z), rotation = ori_rot),
            #  carla.Transform(location = carla.Location(ori_loc.x, ori_loc.y-delta_x, ori_loc.z), rotation = ori_rot),]


vehicle_dict = {}
vehicle_bbox = {}
camera_sensors = {}
semantic_sensors = {}


camera_fov = camera_bp.get_attribute("fov").as_float()
# import pdb; pdb.set_trace()

intrinsic = np.identity(3)
img_x = camera_bp.get_attribute("image_size_x").as_int()
img_y = camera_bp.get_attribute("image_size_y").as_int()
f = img_x / (2 * math.tan(camera_fov * math.pi / 360))

print("camera fov is {} focal length is {}".format(camera_fov, f))
intrinsic[0, 2] = img_x / 2
intrinsic[1, 2] = img_y / 2
intrinsic[0, 0], intrinsic[1, 1] = f, f

def image_callback(image, vehicle_id):
    data = np.frombuffer(image.raw_data, dtype=np.uint8)
    data = np.reshape(data, (image.height, image.width, 4))
    bbox = camera_transform_func( image.transform, image.height, image.width, vehicle_id, image.frame)
    cv2.polylines(data, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)
    if not os.path.exists("./test_out/rgb_with_bbox"):
        os.makedirs("./test_out/rgb_with_bbox/")
    cv2.imwrite("./test_out/rgb_with_bbox/vehicle_{}_{}.png".format(vehicle_id, image.frame), data)
    
# add vehicles
# Spawn 5 vehicles randomly distributed throughout the map 
for i in range(num_vehicle):
    # if i % 10 != 0:
    #     continue
    vehicle = world.try_spawn_actor(random.choice(vehicle_bp), new_trans[i])
    if vehicle == None:
        i -= 1
        continue
    # vehicle.set_autopilot(True)
    vehicle_id = vehicle.id
    vehicle_dict[vehicle_id] = vehicle

    # Create a transform to place the camera on top of the vehicle
    camera_init_trans = carla.Transform(carla.Location(z=height), carla.Rotation(pitch = pitch))

    # We spawn the camera and attach it to our ego vehicle
    # semantic_camera_sensor = world.spawn_actor(semantic_camera_bp, camera_init_trans, attach_to=vehicle)
    
    # We spawn the camera and attach it to our ego vehicle
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    img_callback_with_params = lambda image, vehicle_id = vehicle_id: image_callback(image, vehicle_id)
    # sem_callback_with_params = lambda data, vehicle_id = vehicle_id : semantic_camera_callback(data, vehicle_id)

    camera.listen(img_callback_with_params)
    # semantic_camera_sensor.listen(sem_callback_with_params)

    camera_sensors[i] = camera
    # semantic_sensors[i] = semantic_camera_sensor


# 仿真持续10秒
for _ in range(1):
    world.tick()  # 推进仿真的一个时间步长
    time.sleep(0.1)  # 可选：降低循环速率，避免过度占用CPU

for i, camera in camera_sensors.items():
    camera.stop()

# 摧毁actors
for actor in world.get_actors():
    actor.destroy()
