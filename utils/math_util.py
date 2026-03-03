import numpy as np


"""
这个计算方式看起来是东北天转北西天的转置矩阵T
T=[[0,1,0],
   [-1,0,0],
   [0,0,1]]
旋转矩阵做转置的计算应该是：T(R_ENU)
这里应该是直接认为该矩阵是body2world的矩阵，所以没有做进一步的处理
后面可以尝试调整是不是需要进行转置操作
"""
def quat2rot_change_old(quat):
    w, x, y, z = quat
    r_now = np.zeros(9)
    r_now[0] = 2*x*y + 2*z*w
    r_now[1] = 1.0 - 2*x*x - 2*z*z
    r_now[2] = 2*y*z - 2*x*w
    r_now[3] = 2*y*y + 2*z*z - 1.0
    r_now[4] = 2*z*w - 2*x*y
    r_now[5] = -2*x*z - 2*y*w
    r_now[6] = 2*x*z - 2*y*w
    r_now[7] = 2*y*z + 2*x*w
    r_now[8] = 1.0 - 2*x*x - 2*y*y
    return r_now


def quat2rot_change(quat):
    w, x, y, z = quat
    r_now = np.array([
                        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
                        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
                    ])
    return r_now.T.reshape(-1)

# 这里只是单纯做了一个机体到世界系下的转换
def body2worldVel(r_now, vel):
    # local_vel = np.zeros(3)
    # x, y, z = vel
    R = r_now.reshape(3,3)
    # local_vel[0] = r_now[0]*x + r_now[1]*y + r_now[2]*z
    # local_vel[1] = r_now[3]*x + r_now[4]*y + r_now[5]*z
    # local_vel[2] = r_now[6]*x + r_now[7]*y + r_now[8]*z
    return R @ vel


def errorRot(r_now, r_d):
    # rot = np.zeros(9)
    # rot[0] = r_d[0]*r_now[0] + r_d[3]*r_now[3] + r_d[6]*r_now[6]
    # rot[1] = r_d[0]*r_now[1] + r_d[3]*r_now[4] + r_d[6]*r_now[7]
    # rot[2] = r_d[0]*r_now[2] + r_d[3]*r_now[5] + r_d[6]*r_now[8]
    # rot[3] = r_d[1]*r_now[0] + r_d[4]*r_now[3] + r_d[7]*r_now[6]
    # rot[4] = r_d[1]*r_now[1] + r_d[4]*r_now[4] + r_d[7]*r_now[7]
    # rot[5] = r_d[1]*r_now[2] + r_d[4]*r_now[5] + r_d[7]*r_now[8]
    # rot[6] = r_d[2]*r_now[0] + r_d[5]*r_now[3] + r_d[8]*r_now[6]
    # rot[7] = r_d[2]*r_now[1] + r_d[5]*r_now[4] + r_d[8]*r_now[7]
    # rot[8] = r_d[2]*r_now[2] + r_d[5]*r_now[5] + r_d[8]*r_now[8]
    R_now = r_now.reshape(3,3)
    R_d = r_d.reshape(3,3)
    R_err = R_d.T @ R_now
    return R_err.reshape(-1)


