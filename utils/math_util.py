import numpy as np

def quat2rot_change(r_now, quat):
    w, x, y, z = quat
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


def body2worldVel(r_now, vel):
    local_vel = np.zeros(3)
    x, y, z = vel
    local_vel[0] = r_now[0]*x + r_now[1]*y + r_now[2]*z
    local_vel[1] = r_now[3]*x + r_now[4]*y + r_now[5]*z
    local_vel[2] = r_now[6]*x + r_now[7]*y + r_now[8]*z
    return local_vel


def errorRot(r_now, r_d):
    rot = np.zeros(9)
    rot[0] = r_d[0]*r_now[0] + r_d[3]*r_now[3] + r_d[6]*r_now[6]
    rot[1] = r_d[0]*r_now[1] + r_d[3]*r_now[4] + r_d[6]*r_now[7]
    rot[2] = r_d[0]*r_now[2] + r_d[3]*r_now[5] + r_d[6]*r_now[8]
    rot[3] = r_d[1]*r_now[0] + r_d[4]*r_now[3] + r_d[7]*r_now[6]
    rot[4] = r_d[1]*r_now[1] + r_d[4]*r_now[4] + r_d[7]*r_now[7]
    rot[5] = r_d[1]*r_now[2] + r_d[4]*r_now[5] + r_d[7]*r_now[8]
    rot[6] = r_d[2]*r_now[0] + r_d[5]*r_now[3] + r_d[8]*r_now[6]
    rot[7] = r_d[2]*r_now[1] + r_d[5]*r_now[4] + r_d[8]*r_now[7]
    rot[8] = r_d[2]*r_now[2] + r_d[5]*r_now[5] + r_d[8]*r_now[8]
    return rot
