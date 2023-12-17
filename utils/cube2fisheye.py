#! /usr/bin/python
"""

@author: Miguel Ángel Bueno Sánchez

"""
import numpy as np

import math

FACE_SIZE = 1024

def spherical_to_cartesian(r, phi, fov):
    """
    Transforms spherical coordinates to cartesian
    :param r: matrix with computed pixel heights
    :param phi: matrix with computed pixel angles
    :param fov: desired field of view
    :return: x,y,z cartesian coordinates

    Equidiantant model
    """
    theta = r*fov/2
    # xyz = np.zeros((r.shape[0], r.shape[1], 3))
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    # xyz[:,:,0] = x.astype(int)
    # xyz[:,:,1] = y.astype(int)
    # xyz[:,:,2] = z.astype(int)

    return x, y, z

def get_spherical_coordinates(output_height, output_width):
    """
    基于输出图像的尺寸，确定输出图像上每个点的极坐标(r, phi)
    Finds spherical coordinates on the output image
    :param output_height: height of output image
    :param output_width: width of output image
    :return: two matrices that contain spherical coordinates
    for all pixels of the output image
    """
    cc = (int(output_height/2), int(output_width/2))
    
    y = np.arange(0, output_height, 1)  # 返回一个有终点和起点的固定步长的排列（等差数组）
    x = np.arange(0, output_width, 1)
    
    xx, yy = np.meshgrid(y, x)  # 将y、x中每个数据排列组合生成多个点，将各个点的x坐标放入xx中，y坐标放入yy中
    
    bias = np.ones((output_width, output_height))*cc[0] # 生成元素全为cc[0]的矩阵
    
    xx = np.subtract(xx, bias)  # 将横坐标范围从0~output_height变为+-output_height/2
    yy = np.subtract(yy, bias)
    xx = np.divide(xx,bias)     # 横纵坐标标准化
    xx[:,-1] = 1
    yy = np.divide(yy,-bias)
    yy[-1,:] = -1
    
    r = np.sqrt(xx**2 + yy**2)  # 每个点的极坐标距离，并消除距离大于1的点
    r[r>1] = np.nan
    r[r<0] = 0
    
    phi = np.zeros((output_height, output_width))   # 计算每个点的极坐标角度
    phi[:672,672:] = np.arcsin(np.divide(yy[:672,672:],r[:672,672:]))
    phi[:,:672] = np.pi - np.arcsin(np.divide(yy[:,:672],r[:,:672]))
    phi[673:,672:] = 2*np.pi + \
                     np.arcsin(np.divide(yy[673:,672:],r[673:,672:]))
    phi[cc[0],cc[1]] = 0
    
    return r, phi

def get_face(x, y, z):
    """
    根据三维笛卡尔坐标，确定该点位于立方体哪个表面上
    Finds which face of a cube map a 3D vector with origin
    at the center of the cube points to
    :param x, y, z: cartesian coordinates
    :return: string that indicates the face
    """
    
    max_axis = max(abs(x), abs(y), abs(z))
    
    if math.isclose(max_axis, abs(x)):  # math.isclose 两个数是否绝对/相对接近
        return 'right' if x < 0 else 'left'
    elif math.isclose(max_axis, abs(y)):
        return 'bottom' if y < 0 else 'top'
    elif math.isclose(max_axis, abs(z)):
        return 'back' if z < 0 else 'front'

def raw_face_coordinates(face, x, y, z):
    """
    Finds u,v coordinates (image coordinates) for a given
    3D vector
    :param face: face where the vector points to
    :param x, y, z: vector cartesian coordinates
    :return: uv image coordinates
    """
    
    if face == 'left':
        u = z
        v = -y
        ma = abs(x)
    elif face == 'right':
        u = -z
        v = -y
        ma = abs(x)
    elif face == 'bottom':
        u = -x
        v = -z
        ma = abs(y)
    elif face == 'top':
        u = -x
        v = z
        ma = abs(y)
    elif face == 'back':
        u = x
        v = y
        ma = abs(z)
    elif face == 'front':
        u = -x
        v = -y
        ma = abs(z)
    else:
        raise Exception('Tile ' + face + 'does not exist')
    
    return (u/ma + 1)/2, (v/ma + 1)/2


def create_cubemap(images : dict):
    output_image = np.zeros((3072,3072,3))

    h = images["front"].shape[0] # 1024
    w = images["front"].shape[1] # 1024

    output_image[h:h+h, 0:w] = images["left"]
    output_image[h:h+h, w:w+w] = images["front"]
    output_image[h:h+h, 2*w:2*w+w] = images["right"]
    output_image[0:h, w:w+w] = images["top"]
    output_image[2*h:2*h+h, w:w+w] = images["bottom"]

    return output_image

# 规定每个立方体表面的原点位置
face_origin = {
                'left': (0,FACE_SIZE),
                'front': (FACE_SIZE,FACE_SIZE),
                'right': (2*FACE_SIZE,FACE_SIZE),
                'back': (3*FACE_SIZE,FACE_SIZE),
                'top': (FACE_SIZE,0),
                'bottom': (FACE_SIZE,2*FACE_SIZE),
              }

def tile_origin_coordinates(face):
    """
    Finds the position of each tile on the cube map
    :param face: face where a vector points to
    :return: the position of each tile on the cube map
    """
    return face_origin.get(face)

    # if face == 'left':
    #     return 0, n
    # elif face == 'front':
    #     return n, n
    # elif face == 'right':
    #     return 2*n, n
    # elif face == 'back':
    #     return 3*n, n
    # elif face == 'top':
    #     return n, 0
    # elif face == 'bottom':
    #     return n, 2*n
    # else:
    #     raise Exception('Tile ' + face + 'does not exist')

def normalized_coordinates(face, x, y, n):
    """
    Finds coordinates on the 2D cube map image of a 3D
    vector
    :param face: face where a 3D vector points to
    :param x, y: image coordinates
    :param n: tiles size
    :return: coordinates on the 2D cube map image
    """
    # 首先获取该表面的原点坐标
    tile_origin_coords = tile_origin_coordinates(face)
    
    tile_x = math.floor(x*n)    # math.floor 向下取整函数
    tile_y = math.floor(y*n)
    
    if tile_x < 0:
        tile_x = 0
    elif tile_x >= n:
        tile_x = n-1
    if tile_y < 0:
        tile_y = 0
    elif tile_y >= n:
        tile_y = n-1
    
    x_cubemap = tile_origin_coords[0] + tile_x
    y_cubemap = tile_origin_coords[1] + tile_y

    # if face == 'left' and tile_origin_coords[1] + tile_y == 2048:
    # y_cubemap = 2047
    # else:
    # y_cubemap = tile_origin_coords[1] + tile_y

    return x_cubemap, y_cubemap
