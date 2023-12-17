import yaml
import numpy as np
from utils.geometry import mat_to_trans
from sensor.rgb_camera import turn_intrinsic_to_camera_setting

cam_mat_0 = np.array([[ -0.9999999999999833, -1.8309120287359256e-07, 2.360966110115381e-10,-1.4999966134366787],
                      [1.83091202855184e-07, -0.9999999999999832,  7.044708971986474e-10, 3.5820042398881924e-07],
                      [2.3609648177463916e-10, 7.044709401330534e-10, 0.9999999999999999, 0.3999999443577278],
                      [0,0,0, 1]])

print(mat_to_trans(cam_mat_0))

print(turn_intrinsic_to_camera_setting(np.array([[618.03867197,   0.        , 256.        ],
       [  0.        , 618.03867197, 256.        ],
       [  0.        ,   0.        ,   1.        ]])))