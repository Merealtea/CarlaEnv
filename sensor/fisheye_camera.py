import carla
from utils.geometry import rotation_sub
from utils.cube2fisheye import FACE_SIZE ,spherical_to_cartesian,\
                                 get_spherical_coordinates, create_cubemap,\
                                 get_face, raw_face_coordinates, normalized_coordinates
import numpy as np

FOV = 196
FISHEYE_WIDTH = 1344
FISHEYE_HEIGHT = 1344

def create_fisheye_camera_trans(location, rotation):
    """
        Use cudemap to generate fisheye image, so we need 5 camera to sample rgb data to generate image
        Code from Piaozx
    """
    fisheye_cameras_trans = {}

    rotation_bias = {
        "front" : carla.Rotation(pitch = 0, yaw = 0, roll = 0),
        "top" : carla.Rotation(pitch = -90, yaw = 0, roll = 0),
        "left" : carla.Rotation(pitch = 0, yaw = -270, roll = 0),
        "right" : carla.Rotation(pitch = 0, yaw = -90, roll = 0),
        "bottom" : carla.Rotation(pitch =90, yaw = 0, roll = 0)
    }

    for direction in rotation_bias:
        fisheye_cameras_trans[direction] = carla.Transform(
            location = location,
            rotation = rotation_sub(rotation , rotation_bias[direction])
        )
    return fisheye_cameras_trans


def cubemap_to_fisheye(images: dict):
   
    """
    Converts loaded cube maps into fisheye images
    """
    
    # Create new output image with the dimentions computed above
    output_image = np.zeros((FISHEYE_HEIGHT, FISHEYE_WIDTH,3))
    fov = FOV*np.pi/180
    
    # 确定输出图像平面上每个点的极坐标表达式，返回数组r, phi
    r, phi = get_spherical_coordinates(FISHEYE_HEIGHT, FISHEYE_WIDTH)
    # 将输出图像上的二维极坐标转化为三维空间中对应的空间点坐标
    x, y, z = spherical_to_cartesian(r, phi, fov) 
    cubemap = create_cubemap(images)

    # output_image[np.isnan(r)] = 0
    #-------------------- 将一帧cubemap转换成鱼眼 --------------------#
    for row in range(0, FISHEYE_HEIGHT):
        for column in range(0, FISHEYE_WIDTH):
            if np.isnan(r[row, column]):    # 将输出图像平面上极坐标距离超过1的点设置为黑色
                
                output_image[row, column, 0] = 0
                output_image[row, column, 1] = 0
                output_image[row, column, 2] = 0
            # 对于极坐标距离在1以内的点
            else:
                # 首先确定该点对应的三维坐标指向哪个立方体表面
                face = get_face(x[row, column],
                                y[row, column],
                                z[row, column])
                # 然后确定该点在该立方体表面上的uv坐标
                u, v = raw_face_coordinates(face,
                                            x[row, column],
                                            y[row, column],
                                            z[row, column])
                # 最后获取标准化的uv坐标，锁定输出图像上坐标为(row,column)的点对应着立方体哪个表面上的哪个点
                xn, yn = normalized_coordinates(face,
                                                u,
                                                v,
                                                FACE_SIZE)
                # 将原图像素色彩值转移到输出图像上
                output_image[row, column, :] = cubemap[yn, xn, :]
    #-------------------- 转换完一帧图像后进行存储 --------------------#
    # 将鱼眼图像关于y轴翻转一下子
    output_image = np.flip(output_image, 1)
    return 