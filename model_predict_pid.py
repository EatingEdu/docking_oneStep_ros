import numpy as np
from numpy.linalg import norm
import math

from Force.admittanceVel import *
from Force.estimater import *
from dynamics import *

import pdb

GRAV = 9.81 

def clamp_norm(x, maxnorm):
    n = np.linalg.norm(x)
    return x if n <= maxnorm else (maxnorm / n) * x

def normalize(x):
    n = np.linalg.norm(x)
    if n < 0.00001:
        return x, 0
    return x / n, n

def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

#旋转矩阵转欧拉角
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def quadrotor_jacobian(dynamics):
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T
    torque[2,:] = dynamics.torque_max * dynamics.prop_ccw
    thrust = dynamics.thrust_max * np.ones((1,4))
    dw = (1.0 / dynamics.inertia)[:,None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])
    J_cond = np.linalg.cond(J)
    # assert J_cond < 100.0
    # if J_cond > 50:
    #     print("WARN: Jacobian conditioning is high: ", J_cond)
    return J

class NonlinearPositionController(object):
    def __init__(self, dynamics, force_control=False):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        ## Jacobian inverse for our quadrotor
        # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
        #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
        #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
        #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])
        self.action = None

        #self.kp_p, self.kd_p = 4.5, 3.5
        #self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_p = np.array([4.5, 3.5, 6.4])
        self.kd_p = np.array([4.5, 3.5, 2])
        self.ki_p = np.zeros(3)

        self.kp_a, self.kd_a = 200.0, 70.0 #50.

        #contact_froce 
        self.force_control = force_control
        self.ki_force = 0.1#0.8
        self.e_force = np.zeros(3)

        self.rot_des = np.eye(3)
        # self.rot_des = np.array([[-1.,0.,0.],
        #                         [0.,-1.,0.],
        #                         [0.,0.,1]])

        #self.step_func = self.step

        # pidThrustOmega
        self.angle = np.zeros(3)
        self.last_angle = np.zeros(3)
        self.item_omega = np.zeros(3)
        circle_per_sec = 2* np.pi
        self.angle_p_x = 5.0
        self.angle_p_y = 5.0
        self.angle_p_z = 0.0075#0.008#0.015
        self.kpa = np.array([0.000035,8,0.0075]) #  np.array([0.000035,8,0.0075])
        #self.kda = np.array([0.,8,0.]) # np.array([0.,8,0.])
        self.angle_i = np.zeros(3)
        self.e_pi = np.zeros(3)
        # self.kpa = np.array([20,20,20])
        # self.kda = np.array([7.,7,7])

        max_rp =  0.1 * circle_per_sec
        max_yaw =  0.1 * circle_per_sec
        self.min_omega = np.array([ -max_rp, -max_rp, -max_yaw])
        self.max_omega = np.array([  max_rp,  max_rp,  max_yaw])
        
        #
        self.omega_errlast = None

    def stepThrustOmega50hz(self, pos,vel, rot, omega, goal, mass=1.0):
        self.kp_p = np.array([2.5, 2.5, 18.]) # [1.2 
        self.kd_p = np.array([4.2, 4.2, 9.5]) #np.array([1.2, 3., 9.5])[5.0]
        self.ki_p = np.array([0.008, 0.0008, 0.001])
        
        #20231028
        # self.kp_p = np.array([0.75, 3.8, 18.])
        # self.kd_p = np.array([0.95, 3., 9.5])
        # self.ki_p = np.array([0.001, 0.001, 0.06])
        
        # self.kp_p = np.array([2.7, 4.0, 18.])
        # self.kd_p = np.array([1.5, 3., 9.5]) 
        # self.ki_p = np.array([0.005, 0.005, 0.06])
        
        # self.kpa = np.array([2.5,  2.5, 6.])
        # self.kda = np.array([1.7, 0.7, 0.])
        # self.kpa = np.array([3.,  2.5, 6.])
        # self.kda = np.array([0.8, 0.7, 0.])
        self.kpa = np.array([3.8,  2.5, 6.])
        self.kda = np.array([0.8, 0.7, 0.])

        #self.ki_force = 0.5
        self.ki_force = 0.1

        to_goal = goal - pos
        goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 0.5)
        e_v = vel
        self.e_pi += e_p
        
        #desired force 应该是在机体坐标系下进行描述的 所以需要转换成世界坐标系（需要缕一缕）
        #[Hybrid Force/Motion Control and Internal Dynamics of Quadrotors for Tool Operation] 
        #####################################################################################
        force_item = np.zeros(3)
        # self.e_force += contact_force - desired_force_world
        # force_item = desired_force_world - self.ki_force * self.e_force * dt #dynamics.contact_torque
        # force_item[0] = 0.
        # force_item[2] = 0.
        # if not self.force_control or (not np.any(contact_force)): #判断是否接触力为0
        #     force_item[1] = 0.
            
        #acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV]) - force_item/mass
        acc_des = -self.ki_p * self.e_pi - self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV]) - force_item/mass

        
        xc_des = self.rot_des[:, 0] 

        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = rot
        #self.angle = rotationMatrixToEulerAngles(R)

        # des_omega = (self.angle - self.last_angle)/dt
        #print(R)
        def vee(R):
            return np.array([R[2,1], R[0,2], R[1,0]])
        e_R = vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2 # slow down yaw dynamics
        e_w = omega
        

        #dw_des = -self.kp_a * e_R - self.kd_a * e_w
        
        thrust_mag = np.dot(acc_des, R[:,2])

        item_thrust = thrust_mag / GRAV - 1

        dw_des_2 = -self.kpa * e_R - self.kda * e_w
        action = [item_thrust, dw_des_2[0], dw_des_2[1], dw_des_2[2]]
        new_action = np.array(action)
        return new_action, _
    
    def step_force_torque(self,  dynamics, goal, dt, action=None, flag="body",observation=None):
        # kp = random.randint(9,11) # could be more aggressive
        #pdb.set_trace()
        # pdb.set_trace()
        # kp =  random.uniform(9,13)
        #print("thrustOmega")
        kp = 10
        ki = 2
        kd = 0.0
        kff = 0
        kpp = 10
        kpi = 0
        kpd = 0
        kpff = 0
        krp = 10
        kri = 0
        krd = 0
        krff = 0
        kyp = 10
        kyi = 0
        kyd = 0
        kyff = 0
        detla_time = 0.05 #0.01
        # dynamics.omega = action[1:]
        # action[0] = 0.5
        # action[1] = 2 * np.pi
        # action[2] = 2 * np.pi
        # action[3] = 1 * np.pi
        
        #pdb.set_trace()
        #bc train 20230618
        ######################
        # thrust = action[0]
        # action = action * 0.8
        # action[0] = thrust
        ###########################
        #action[0] -= 0.5

        omega_err = dynamics.omega - action[1:]
        i_factor = np.array([0.,0.,0.])
        i_factor[0] = omega_err[0]/7
        i_factor[1] = omega_err[1]/7
        i_factor[2] = omega_err[2]/7
        d_input = (omega_err - (self.omega_errlast if (self.omega_errlast is not None) else omega_err))/dt
        self.omega_errlast = omega_err 
        # omega_err = np.array([1.0,1.0,1.0])
        # err_omega_integral_x = self.integrator(err_omega[0],0,err_omega_integral_ls,detla_time,0)
        # err_omega_integral_y = self.integrator(err_omega[1],0,err_omega_integral_ls,detla_time,1)
        # err_omega_integral_z = self.integrator(err_omega[2],0,err_omega_integral_ls,detla_time,2)
        dynamics.omega_errls[0] += omega_err[0] * detla_time * (1 - i_factor[0]*i_factor[0])
        dynamics.omega_errls[1] += omega_err[1] * detla_time * (1 - i_factor[1]*i_factor[1])
        dynamics.omega_errls[2] += omega_err[2] * detla_time * (1 - i_factor[2]*i_factor[2])
        dw_des = -kp * omega_err - ki * dynamics.omega_errls + kd * d_input - kff * action[1:]
        # dw_des = -kp * omega_err
        # action[0] = dynamics.thrust_to_weight - 1.0
        acc_des = GRAV * (action[0] + 1.0) 
        
        # rnd = np.random.normal(loc=0.0, scale=0.6,size=1)
        # acc_des = acc_des + rnd
        #acc_des = action[0] / dynamics.mass
        #acc_des = (action[0] +1) / dynamics.mass - GRAV
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        # pdb.set_trace()
        # vvfin = np.matmul(self.vv, thrusts)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        force, torque = dynamics.step(thrusts, dt, flag)
        return force, torque
    
      
    def reset(self,):
        self.action = None

        #self.kp_p, self.kd_p = 4.5, 3.5
        #self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_p = np.array([4.5, 3.5, 6.4])
        self.kd_p = np.array([4.5, 3.5, 2])
        self.ki_p = np.zeros(3)

        self.kp_a, self.kd_a = 200.0, 70.0 #50.

        #contact_froce 
        self.ki_force = 0.1#0.8
        self.e_force = np.zeros(3)

        self.rot_des = np.eye(3)
        # self.rot_des = np.array([[-1.,0.,0.],
        #                         [0.,-1.,0.],
        #                         [0.,0.,1]])

        #self.step_func = self.step

        # pidThrustOmega
        self.angle = np.zeros(3)
        self.last_angle = np.zeros(3)
        self.item_omega = np.zeros(3)
        circle_per_sec = 2* np.pi
        self.angle_p_x = 5.0
        self.angle_p_y = 5.0
        self.angle_p_z = 0.0075#0.008#0.015
        self.kpa = np.array([0.000035,8,0.0075]) #  np.array([0.000035,8,0.0075])
        #self.kda = np.array([0.,8,0.]) # np.array([0.,8,0.])
        self.angle_i = np.zeros(3)
        self.e_pi = np.zeros(3)
        # self.kpa = np.array([20,20,20])
        # self.kda = np.array([7.,7,7])

        max_rp =  0.1 * circle_per_sec
        max_yaw =  0.1 * circle_per_sec
        self.min_omega = np.array([ -max_rp, -max_rp, -max_yaw])
        self.max_omega = np.array([  max_rp,  max_rp,  max_yaw])


# global dynamics_uav1,   dynamics_uav2,
# global pid_control_uav1, pid_control_uav1
dynamics_uav1 = Dynamics(thrust_to_weight=1/0.5)  
dynamics_uav2 = Dynamics(thrust_to_weight=1/0.5)  


pid_control_uav1 = NonlinearPositionController(dynamics_uav1)
pid_control_uav2 = NonlinearPositionController(dynamics_uav2)
# global first, rt
# first = True
#rt = init_estimate_force_torque(mass=1.6)
dt = 0.01
def modelPredict_pid(uav1, uav2, rot1, rot2, rt):
    #global first
    dynamics_uav1.update_state(uav1.pos, uav1.vel, uav1.omega, rot1)
    action1, _ = pid_control_uav1.stepThrustOmega50hz(uav1.pos, uav1.vel, rot1, uav1.omega, uav1.nominal_pos , uav1.mass)
    action2, _ = pid_control_uav2.stepThrustOmega50hz(uav2.pos, uav2.vel, rot2, uav2.omega, uav2.nominal_pos , uav2.mass)
    
    force_torque = pid_control_uav1.step_force_torque(dynamics=dynamics_uav1, goal=uav1.nominal_pos, dt=dt, action=action1)
    #force_torque = pid_control_uav1.step_force_torque(dynamics=dynamics, goal=goal, dt=dt, action=action)
    
    estimate_force_torque = get_estimate_force_torque(rt, uav1, rot1, force_torque)
    return np.concatenate([action1, action2]), estimate_force_torque



        
def get_estimate_force_torque(rt, uav, rot, force_torque):
    rt.u = np.array([0.,0.,force_torque[0]])
    rt.torq_b = np.array(force_torque[1:]).reshape(-1,1)
    rt.v = uav.vel.reshape(-1,1)
    rt.R_b = rot
    rt.w_vel = uav.omega.reshape(-1,1)
    estimate_force_torque = rt.getRt()
    estimate_force_world = estimate_force_torque.reshape(1,-1)[0]
    estimate_force_world[2] = -estimate_force_world[2]
    #estimate_force_pub.publish(Float64MultiArray(data=estimate_force_world))
    return estimate_force_world
    
