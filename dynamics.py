import numpy as np
import sys
from utils.quad_utils import *
GRAV = 9.81
EPS = 1e-6
import copy

class Dynamics():
    def __init__(self, mass=1.7795, 
                    thrust_to_weight= 2.941,
                    torque_to_thrust = 0.0104,
                    sense_noise = None,
                    motor_assymetry = np.ones(4),
                    motor_damp_time_up = 0.2,
                    motor_damp_time_down = 0.15,
                    prop_pos = np.array([[0.1591    , -0.1591    ,  0.2166],
                                        [-0.1591   , -0.1591   , 0.2166],
                                        [-0.1591   , 0.1591    ,  0.2166],
                                        [0.1591    ,  0.1591    ,  0.2166]]),
                    prop_ccw = np.array([-1.,  1., -1.,  1.]), #[-1.,  1., -1.,  1.]
                    inertia = np.array([0.00913226, 0.00913226, 0.0175858]),
                    pos = np.zeros(3),
                    vel = np.zeros(3),
                    omega = np.zeros(3),
                    rot = np.eye(3),
                    desired_force=np.zeros(3)):

        self.mass = mass
        self.inertia = inertia
        self.motor_assymetry = motor_assymetry
        self.thrust_to_weight = thrust_to_weight
        self.torque_to_thrust = torque_to_thrust
        self.prop_pos = prop_pos
        
        self.prop_ccw = prop_ccw
        self.motor_damp_time_up = motor_damp_time_up
        self.motor_damp_time_down = motor_damp_time_down
        self.thrust_cmds_damp = np.zeros(4)
        self.thrust_rot_damp = np.zeros(4)
        self.motor_linearity = 1.0
        
        #thrust_noise
        self.thrust_noise_ratio = 0.05
        self.thrust_noise = OUNoise(4, sigma=0.2*self.thrust_noise_ratio)

        #sense_noise
        #self.update_sense_noise(sense_noise)

        #pdb.set_trace()
        self.prop_crossproducts = self.get_prop_crossproducts()
        self.thrust_max = self.get_thrust_max()
        self.torque_max = self.get_torque_max()

        self.pos = pos
        self.vel = vel
        self.omega = omega
        self.rot = rot
        self.first_time = copy.deepcopy(pos)
        self.pre_dist = copy.deepcopy(pos)
        self.int_dist = 0.

        #observation_space
        self.omega_max = 40. #rad/s The CF sensor can only show 35 rad/s (2000 deg/s), we allow some extra
        self.vxyz_max = 3.

        #thrust_omega
        self.omega_errls = np.array([0.0,0.0,0.0])

        #contact_force
        self.contact_force = np.array([0., 0., 0.])
        self.contact_torque = np.zeros(3)
        self.desired_force = desired_force
        self.desired_force_world = np.zeros(3)

    def get_prop_crossproducts(self,):
        return np.cross(self.prop_pos, np.array([0., 0., 1.]))

    def get_thrust_max(self,):
        return GRAV * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0 # 12.836

    def get_torque_max(self,):
        return self.torque_to_thrust * self.thrust_max
        # change by code_env 20230613 
        #self.torque_max = np.array([0.055562, 0.055562, 0.055562, 0.055562])
        #return self.torque_max

    def update_state(self,pos, vel, omega, rot):
        self.pos = pos
        self.vel = vel
        self.omega = omega
        self.rot = rot

    def angvel2thrust(self, w, linearity=0.424):
        """
        Args:
            linearity (float): linearity factor factor [0 .. 1].
            CrazyFlie: linearity=0.424
        """
        return  (1 - linearity) * w**2 + linearity * w

    def update_sense_noise(self, sense_noise):
        #pdb.set_trace()
        if isinstance(sense_noise, dict):
            self.sense_noise = SensorNoise(**sense_noise)
        elif isinstance(sense_noise, str):
            if sense_noise == "default":
                self.sense_noise = SensorNoise(bypass=False)
            elif sense_noise == "self_define":
                self.sense_noise = SensorNoise(pos_norm_std=0., pos_unif_range=0., 
                        vel_norm_std=0.02, vel_unif_range=0., 
                        quat_norm_std=0.002, quat_unif_range=0., 
                        omega_norm_std=0.06, omega_unif_range=0.,bypass=False,
                        acc_static_noise_std=0, acc_dynamic_noise_ratio=0.005)
            else:
                ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))
        elif sense_noise is None:
            self.sense_noise = SensorNoise(bypass=True)
        else:
            raise ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))

    def step(self,thrust_cmds,dt, flag="body"):
        #print(f"thrust_cmds is {thrust_cmds}")
        thrust_cmds = np.clip(thrust_cmds, a_min=0., a_max=1.)
        self.motor_tau_up = 4*dt/(self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4*dt/(self.motor_damp_time_down + EPS)
        motor_tau = self.motor_tau_up * np.ones([4,])
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down 
        motor_tau[motor_tau > 1.] = 1.

        thrust_rot = thrust_cmds**0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp       
        self.thrust_cmds_damp = self.thrust_rot_damp**2

        ## Adding noise
        thrust_noise = thrust_cmds * self.thrust_noise.noise()
        self.thrust_cmds_damp = np.clip(self.thrust_cmds_damp + thrust_noise, 0.0, 1.0)        
        #pdb.set_trace()
        thrusts = self.thrust_max * self.angvel2thrust(self.thrust_cmds_damp, linearity=self.motor_linearity)
        
        #trans 4 thrust
        # return 
        
        #Prop crossproduct give torque directions
        torques = self.prop_crossproducts * thrusts[:,None] # (4,3)=(props, xyz)

        # additional torques along z-axis caused by propeller rotations
        #pdb.set_trace()
        torques[:, 2] += self.torque_max * self.prop_ccw * self.thrust_cmds_damp   #四个旋翼的反力矩
        torque = np.sum(torques, axis=0) #np.sum(torques, axis=1)
        #print(f"四个反力矩 {self.torque_max * self.prop_ccw * self.thrust_cmds_damp}")
        thrust = np.sum(thrusts)
        if flag == "prop":
            #pdb.set_trace()
            thrusts = self.thrust_max * thrust_cmds + self.torque_max * self.prop_ccw * self.thrust_cmds_damp 
            #print(f"四个拉力值 {thrusts}")
            return thrusts, torque#np.sum(torques, axis=1)
        elif flag == "body":
            #print(f"四个拉力值 {thrusts}")
            return thrust, torque

    # def step2(self,thrust_cmds,dt, flag="body"):
    #     thrust1, torque1 = self.step(thrust_cmds, dt, flag="body")
    #     thrust2, torque2 = self.step(thrust_cmds, dt, flag="body")
    #     return [thrust1,thrust2] , [torque1,torque2]

    # def step1(self,thrust_cmds,dt, flag="body"):
    #     thrust1, torque1 = self.step(thrust_cmds, dt, flag="body")
    #     return [thrust1] , [torque1]
