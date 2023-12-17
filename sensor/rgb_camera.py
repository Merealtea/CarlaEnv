import carla
import numpy as np
import cv2
from actor.vehicle import bounding_box
from os.path import join

def get_intrinsic_matrix(camera_bp):
    intrinsic = np.identity(3)
    
    camera_fov = camera_bp.get_attribute("fov").as_float()
    img_x = camera_bp.get_attribute("image_size_x").as_int()
    img_y = camera_bp.get_attribute("image_size_y").as_int()
    f = img_x / (2 * np.tan(camera_fov * np.pi / 360))

    intrinsic[0, 2] = img_x / 2
    intrinsic[1, 2] = img_y / 2
    intrinsic[0, 0], intrinsic[1, 1] = f, f
    return intrinsic, img_x, img_y

def turn_intrinsic_to_camera_setting(intrinsic):
    img_x, img_y = int(intrinsic[0, 2] * 2), int(intrinsic[1, 2] * 2)
    f = intrinsic[0, 0]
    fov = np.arctan(2 * f / img_x) * 360 / np.pi
    return img_x, img_y, fov


def turn_world_cood_to_camera_coord(points, camera_trans, intrinsic):
    # Points shape [4, n]

    # This (4, 4) matrix transforms the points from world to sensor coordinates.
    world_2_camera = np.array(camera_trans.get_inverse_matrix())

    # Transform the points from world space to camera space.
    points = np.dot(world_2_camera, points)

    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    points = np.array([
        points[1],
        points[2]* -1,
        points[0]])

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points = np.dot(intrinsic, points)

    # Remember to normalize the x, y values by the 3rd value.
    points = np.array([
        points[0, :] / points[2, :],
        points[1, :] / points[2, :]]).astype(int).T
    return points


def turn_camera_cood_to_world_coord(pixel_point, camera_trans, intrinsic, depth):
    # Pixel Points shape [2, n]
    # pixel_point = np.array([pixel_point[0], pixel_point[1], np.ones((len(pixel_point[0]), ))])
    # pixel_point = np.dot(np.linalg.inv(intrinsic), pixel_point)

    pixel_point = np.array([(pixel_point[0] - intrinsic[1,2]).astype(float) * depth / intrinsic[1,1],
                    (pixel_point[1] - intrinsic[0,2]).astype(float) * depth / intrinsic[0,0],
                    depth])

    # dist = np.sqrt(np.sum(pixel_point ** 2, axis = 0, keepdims =True))
    # pixel_point = pixel_point * depth / dist

    pixel_point = np.array([
        pixel_point[2],
        pixel_point[0],
        pixel_point[1] * -1,
        np.ones((len(pixel_point[0],)))
        ])

    # This (4, 4) matrix transforms the points from world to sensor coordinates.
    camera_2_world = np.array(camera_trans.get_matrix())

    # Transform the points from world space to camera space.
    pixel_point = np.dot(camera_2_world, pixel_point)
    pixel_point = pixel_point[:2].T
    return pixel_point


def create_surrond_camera_trans():
    camera_trans = {}
    
    # Front view
    camera_trans[0] = carla.Transform(
        carla.Location(2.55, 0, 0.7),
        carla.Rotation(pitch = 0, yaw = 0, roll = 0)
    )

    # Left view
    camera_trans[1] = carla.Transform(
        carla.Location(x = -0.20861, y = 0.944519, z = 0.1),
        carla.Rotation(pitch = 0, yaw =1.745329 * 180 / np.pi, roll = 0)
    )

    # Right view
    camera_trans[2] = carla.Transform(
        carla.Location(x= 0.20861, y = -0.944471, z = 0.1),
        carla.Rotation(pitch=0, yaw=-1.745330 * 180 / np.pi, roll=0)
    )

    # Back view
    camera_trans[3] = carla.Transform(
        carla.Location(x=-2.5, y=0, z = 0.4),
        carla.Rotation(pitch=-0, yaw=3.141593 * 180 / np.pi, roll=0)
    )

    return camera_trans    
