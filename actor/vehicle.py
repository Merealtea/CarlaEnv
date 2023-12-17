import numpy as np

def bounding_box(extend, transform):
    x, y, z = extend.x, extend.y, 0
    box = np.array([[-x, y, z], [x, y, z], [x, -y, z], [-x, -y, z]]).T
    center = np.array([[transform.location.x, transform.location.y, transform.location.z]]).T
    yaw = transform.rotation.yaw / 180 * np.pi
    box = np.array([[np.cos(yaw), -np.sin(yaw), 0],[np.sin(yaw), np.cos(yaw), 0], [0,0,1]]).dot(box)+ center
    # box + center #
    box = np.concatenate([box, np.ones((1, box.shape[1]))], axis = 0)
    return box

