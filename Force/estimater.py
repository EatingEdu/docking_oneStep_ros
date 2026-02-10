import numpy as np
import math
import pdb

class EnvForce():
    def __init__(self, num, interval, force_type="static", torque_flag = False,f_x=0, f_y =0, f_z =0):
        self.num = num
        self.interval = interval
        self.force_type = force_type
        self.steps = 1000
        self.x =  np.linspace(0, 4 * np.pi, self.steps)
        # self.y =  0.1* np.linspace(0, 2 * np.pi, 4010)
        self.f_x = f_x
        self.f_y = f_y
        self.f_z = f_z
        self.f = np.array([0,0,0])
        self.torque = np.array([0,0,0])
        self.last_f = self.f
        self.last_torque = self.torque
        self.expect_pos = np.array([0,-0.25,2])
        #self.update(num, interval)
        self.init_distance = 0.02
        self.k_wall = 300
        self.k_mu = 0.1
        self.torque_flag = torque_flag
        # is random
        self.r_flag = False

    
    #scene1 后坐力的仿真
    def ImpulseFunc(self, num, interval):
        if num % interval < interval/150:
            self.f = np.array([0.,5.,0.])
            self.torque = np.zeros(3)
        else:
            self.f = np.zeros(3)
            self.torque = np.zeros(3)

    #scene 2 fix force
    def staticForce(self, num, interval):
        #static force
        #pdb.set_trace()
        if self.r_flag: 
            if num % interval > interval/2:
                self.f = np.array([0.,0.,0.])
                self.torque = np.array([0.,0.,0.])
            else:
                self.f = self.last_f
                self.torque = self.last_torque
        else:
            self.r_flag = True
            self.randomFunction(num, interval)
            self.last_f = self.f
            self.last_torque = self.torque

    #scene 2 fix force
    def staticStableForce(self, num, interval):
        #static force 
        if num % interval < interval/2:  #default F450 1.2  crazyflie:0.8
            self.f = np.array([0.,1.2, 0.])
            self.torque = np.array([0.,0.,0.])
        else:
            self.f = np.zeros(3)
            self.torque = np.zeros(3)

        # self.torque = np.array([0.1*self.f_x, 0.1*self.f_y, 0.1*self.f_z])
        # self.f = np.array([0,1.4,0])

    #scene3 move along the wall
    #ps:外力与外扭矩都是body系下的，位置与速度都是world下，所以f_y与f_x都是world系下
    #[1] Reshaping the Physical Properties of a Quadrotor through IDA-PBC and its Application to Aerial Physical Interaction
    def moveAlongTheWall(self, vel=None, rot=None, real_pos=None):
        #pdb.set_trace()
        if str(real_pos) == "None" :
            f_y = k_wall * self.init_distance
            rot = np.eye(3)
            vel = np.zeros(3)
        elif real_pos[1] >= 0:
            self.f = np.zeros(3)
            self.torque = np.zeros(3)
            return None
        else:
            #pdb.set_trace()
            f_y = self.k_wall * np.abs(real_pos[1])#弹簧形变产生的力,初始化有一个固定值，后面需要根据实际值的差进行变化
            f_x = -self.k_mu * vel #摩擦力
            d = np.array([0, 0.25, 0]) #位置差
            f = np.array([0, f_y,0]) + f_x
            # print(f"f_y is {f_y}")
            # print(f"f_x is {f_x}")
            # print(f"real_pos[1] is {real_pos[1]}")
        self.f = rot.T @ f
        self.torque = np.cross(d, rot.T) @ f
        # self.f = np.zeros(3)
        # self.torque = np.zeros(3)

    def stepFunction(self,num,  interval):
        if num % interval < interval/2:
            self.f = np.array([2,3,5])
            self.torque = np.array([self.f_x,self.f_y,self.f_z])
        else:
            self.f = np.array([2,3,5])
            self.torque = np.array([0., 0., 0.])

    def randomFunction(self,num, interval):
        # f_ = np.array(self.f_x, self.f_y, self.f_z)
        self.f = np.array([np.random.uniform(0,self.f_x) , np.random.uniform(0,self.f_y),np.random.uniform(0,self.f_z)])
        self.torque = np.array([np.random.uniform(0,self.f_x) , np.random.uniform(0,self.f_y),np.random.uniform(0,self.f_z)])
    
    # def randomStaticFunction(self,num, interval):
    #     self.f = np.array([np.random.uniform(0,self.f_x) , np.random.uniform(0,self.f_y),np.random.uniform(0,self.f_z)])
    #     self.torque = np.array([np.random.uniform(0,self.f_x) , np.random.uniform(0,self.f_y),np.random.uniform(0,self.f_z)])

    def cosFunction(self, num, interval):
        #pdb.set_trace()
        self.last_f = self.f
        self.last_torque = self.torque
        f_ =  np.cos(self.x[num % self.steps])
        self.f = np.array([self.f_x * f_ ,self.f_y * f_ ,self.f_z*f_]) #+ np.random.uniform(0,1)
        t = 0.1 * np.cos(self.x[num % self.steps])
        self.torque = np.array([t,t,t]) #+ np.random.uniform(0,0.1)
        #self.torque = np.array([0., 0., 0.])

    def update(self, num, interval, others):
        if self.force_type == "cos":
            self.cosFunction(num, interval)
        elif self.force_type == "step":
            self.stepFunction(num, interval)
        elif self.force_type == "random":
            self.randomFunction(num, interval)
        elif self.force_type == "impulse":
            self.ImpulseFunc(num, interval)
        elif self.force_type == "static":
            self.staticForce(num, interval)
        elif self.force_type == "staticStable":
            self.staticStableForce(num, interval)
        elif self.force_type == "wall_slide":
            self.moveAlongTheWall(others["vel"], others["rot"], others["real_pos"])
        else:
            #static force
            self.torque = np.zeros(3)
            self.f = np.zeros(3)
        if not self.torque_flag:
            self.torque = np.zeros(3)

class RT():
    def __init__(self, v, w_vel, u, R_b, torq_b, K1, K1_K2, I_b, dt = 0.005,m = 1.5):
        self.dt = dt
        self.I_b = I_b
        self.m = m
        self.g = np.array([0,0,-9.81]).reshape(-1,1)

        self.v = np.array([0,0,0]).reshape(-1,1)
        self.w_vel = w_vel
        self.R_b = R_b
        self.torq_b = torq_b
        self.u = u
        
        self.K1 = K1
        self.K1_K2 = K1_K2

        # self.LastA()
        self.last_A = np.zeros(6).reshape(-1,1)
        self.last_r = np.zeros(6).reshape(-1,1)
        self.last_q = np.zeros(6).reshape(-1,1)
        self.last_double_rA = np.zeros(6).reshape(-1,1)
        self.r_integ = np.zeros(6).reshape(-1,1)
        self.q_integ = np.zeros(6).reshape(-1,1)
        self.A_integ = np.zeros(6).reshape(-1,1) #self.last_A  #
        self.double_rA_integ = np.zeros(6).reshape(-1,1) #self.r_integ + self.A_integ*dt #


    def cross(self,x):
        return np.array([0, -x[2][0], x[1][0], x[2][0], 0, -x[0][0], -x[1][0], x[0][0], 0]).reshape(3,3)

    def LastA(self):#thrust 1d, g = [0,0,G].T
        A_0 = (self.R_b @ self.u).reshape(-1,1) +  self.m  * self.g
        A_1 = self.torq_b - self.cross(self.w_vel) @ (self.I_b @ self.w_vel)
        self.last_A = np.concatenate([A_0,A_1])

    def AInteg(self):
        self.A_integ += self.last_A*self.dt

    def LastQ(self):#
        M_q_0 = self.m * self.v
        M_q_1 = self.I_b @ self.w_vel
        M_q = np.concatenate((M_q_0, M_q_1), axis = 0)
        self.last_q = M_q

    def LastDoubleRA(self):
        self.last_double_rA = self.r_integ + self.A_integ

    def RInteg(self):
        self.r_integ += self.last_r * self.dt

    def QInteg(self):
        self.q_integ += self.last_q * self.dt 

    def DoubleRAInteg(self,): 
        self.double_rA_integ += self.last_double_rA * self.dt

    #final r_t
    def getRt(self):
        self.r_t = -self.K1 @ self.r_integ + self.K1_K2 @ (self.q_integ - self.double_rA_integ)
        self.last_r = self.r_t
        self.LastQ()
        self.LastA()
        self.RInteg()
        self.QInteg()
        self.AInteg()
        self.LastDoubleRA()
        self.DoubleRAInteg()
        return self.r_t
        
def getK(w,epsilon):
    K1_K2 = w**2 * np.eye(6)
    K1 = 2*epsilon*w *np.eye(6)
    return K1, K1_K2


def main():
    rot = np.array( [[ 0.75330233, -0.65762393, -0.00814679],
            [ 0.65766085,  0.75330851 , 0.00291395],
            [ 0.00422077 ,-0.00755291 , 0.99996257]])
    dt = 0.005
    w = np.ones(6) * 0.005
    epsilon  = np.ones(6)
    K1,K1_K2 = getK(w,epsilon)

    I_b = np.array([[0.01340, 0, 0], [0, 0.01340, 0],[0, 0, 0.02389]])
    u = 0.05580946
    v = np.zeros(3).reshape(-1,1)
    w_vel = np.zeros(3).reshape(-1,1)
    
    torq_b = np.array([-1.03745597e-04,  1.05859949e-04,  3.77220161e-05]).reshape(-1,1)
    t = 1
    q_0 = np.zeros(6).reshape(-1,1)
    r_0 = np.zeros(6).reshape(-1,1)
    m = 1.5
    #v, w_vel, u, R_b, torq_b, K1, K1_K2, I_b, dt = 0.005,m = 1.5):
    
    rt = RT(v, w_vel, u, rot, torq_b, K1, K1_K2, I_b)
    for i in range(5):
        res = rt.getRt()
        print(f"finish!!!!!!!!!!!!!!!!!!!!!!!!{i}+  {res}")


if __name__ == "__main__":
    main()
