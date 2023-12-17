import numpy as np
import carla

def mat_to_trans(mat):
    """
        mat is a 4 x 4 mat
        mat[:3,:3] is the rotation matrix
        mat[:3, 4] is the transform 

    """

    trans = carla.Transform(
        carla.Location(mat[0, 3], mat[1, 3], mat[2, 3]),
        carla.Rotation(
            pitch = np.arctan2(- mat[2, 0], np.sqrt(mat[2, 1]**2 + mat[2, 2]**2)),
            yaw = np.arctan2(mat[1, 0], mat[0, 0]),
            roll = np.arctan2(mat[2, 1], mat[2, 2])
        )
    )
    return trans

def turn_trans_to_list(trans):

    return [
        trans.location.x,
        trans.location.y,
        trans.location.z,
        trans.rotation.pitch,
        trans.rotation.yaw,
        trans.rotation.roll,

    ]

def turn_extent_to_list(extent):
    return[
        extent.x,
        extent.y,
        extent.z
    ]

def turn_velocity_to_speed(velocity):
    return np.sqrt(
        velocity.x**2+ velocity.y**2 + velocity.z**2
    ).tolist()

def rotation_sub(src_rot, des_rot):
    return carla.Rotation(
        yaw = src_rot.yaw - des_rot.yaw,
        pitch = src_rot.pitch - des_rot.pitch,
        roll = src_rot.roll - des_rot.roll
    )

def location_sub(src_loc, des_loc):
    return carla.Location(
        x = src_loc.x - des_loc.x,
        y = src_loc.y - des_loc.y,
        z = src_loc.z - des_loc.z
    )

def get_map_center(carla_map):
    sps = carla_map.get_spawn_points()
    sps = np.array([(point.location.x, point.location.y) for point in sps])

    center_location = carla_map.get_spawn_points()[10].location  # 这里使用地图的一个spawn point作为中心点
    center_coords = (center_location.x, center_location.y)

    # # 获取地图尺寸（在x和y轴上的距离）
    # map_bounds = carla_map.get_bounds()
    # map_size_x = map_bounds.x
    # map_size_y = map_bounds.y

    return center_coords#, map_size_x, map_size_y