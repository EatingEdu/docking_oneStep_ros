import numpy as np
import random
import pdb
#import matplotlib.pyplot as plt

class Admittance_rk():
    def __init__(self,
                    w = np.ones(6)*0.5,
                    zita = np.ones(6)*0.8,
                    k1=np.ones(18).reshape(6,3) , 
                    k2 = 0.5* np.ones(18).reshape(6,3), 
                    k3 = 0.5 * np.ones(18).reshape(6,3), 
                    F = 0.5 * np.ones(6).reshape(6,1), 
                    x_n = np.zeros(6).reshape(6,1), 
                    y_n = np.zeros(6).reshape(6,1), 
                    z_n = np.zeros(6).reshape(6,1), 
                    h = 0.1 * np.zeros(6).reshape(6,1), 
                    dt = 0.005, 
                    stable=False 
                    ): 
        self.w = w 
        self.zita = zita 
        
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.F = F
        self.x_n = x_n
        self.h = h
        self.z_n = z_n
        self.y_n = y_n
        self.acc = 0
        self.stable = stable
        self.param_change = False
        self.update_parameter(self.w,self.zita)
        
        # self.dt = dt
        # self.y_last = y_n
        # self.y_n = y_n
        # self.vel = y_n
        # self.last_vel = self.vel
        # self.dt = dt

    def func_zx(self, x, y, z):
        #res = np.transpose(1/self.k1) @ (self.F - self.k3 @ y - self.k2 @ z)
        res = self.F - self.k3 @ y - self.k2 @ z
        return res
        #return np.exp(2*x)*np.sin(x) - 2 * y + 2 * z

    def func_yz(self, y, z):
        return z

    #四阶龙格库塔解二阶微分方程
    def control_rk_2(self,):
        # print(f"self.k2 is {np.diagonal(self.k2)} self.k3 is {np.diagonal(self.k3)}")
        self.last_yn = self.y_n
        self.last_zn = self.z_n
        self.last_acc = self.acc
        self.x_n += 1
        k11 = self.func_yz(self.y_n, self.z_n)
        k21 = self.func_zx(self.x_n, self.y_n, self.z_n)
        
        k12 = self.func_yz(self.y_n + self.h/2, self.z_n + self.h/2 * k21)
        k22 = self.func_zx(self.x_n + self.h/2, self.y_n + self.h/2 * k11, self.z_n + self.h/2 * k21)
        
        k13 = self.func_yz(self.y_n + self.h/2, self.z_n + self.h/2 * k22)
        k23 = self.func_zx(self.x_n + self.h/2, self.y_n + self.h/2 * k12, self.z_n + self.h/2 * k22)
        
        k14 = self.func_yz(self.y_n + self.h/2, self.z_n + self.h * k23)
        k24 = self.func_zx(self.x_n + self.h, self.y_n + self.h*k13, self.z_n + self.h*k23)
        
        self.y_n = self.y_n + self.h/6 * (k11 + 2*(k12 + k13) + k14)
        self.z_n = self.z_n + self.h/6 * (k21 + 2*(k22 + k23) + k24)
        # print(self.x_n , self.y_n, self.z_n)

        #overflow excute clip instead 
        # check = self.y_n.reshape(1,6)[0]
        # a = [i for i in check > 100 if i]
        # b = [i for i in check < -100 if i]
        # if len(a) >0 or len(b) >0 :
        #     print("a or b exception")
        #     self.y_n = np.zeros(6).reshape(-1,1)
        #     self.z_n = np.zeros(6).reshape(-1,1)

        # #velocity_error
        # self.last_vel = self.vel
        # self.vel = (self.y_n - self.y_last) / self.dt

        # #acc_error
        # print(f"the y_n is {self.y_n}")
        # print(f"the z_n is {self.z_n}")
        # print(f"the acc is {self.acc}")
        self.acc = self.func_zx( self.x_n, self.y_n, self.z_n)
        # self.y_n = np.clip(self.y_n, a_min=-10., a_max=10.)
        # self.z_n = np.clip(self.z_n, a_min=-10., a_max=10.)
        # self.acc = np.clip(self.acc, a_min=-10., a_max=10.)
        return self.y_n, self.z_n, self.acc

    
    def update_parameter(self, w, zita, n=6):
        #single train
        # w_ = 0.6
        # print(f"the net w is {w} ,the net zita is {zita}")
        # w_2 = 5
        # # zita_ = 0.2
        # zita_2 = 0.05
        # w = np.array([ w_2, w_2, w_2,w[0], w[1], w[2]])
        # zita = np.array([ zita_2, zita_2, zita_2,zita[0], zita[1], zita[2] ])
        
        # pdb.set_trace()
        # print(f"the net w is {w} ,the net zita is {zita}")
        #print(f"self.stable is {self.stable}")
        if self.stable == "test":
            #max
            self.w = np.ones(6)*3
            self.zita = np.array([1.,1.,1.,1.,1.,1.])
        elif self.stable == "max":
            #max
            self.w = np.array([20,20,20,20,20,20])
            self.zita = np.array([1.,1.,1.,1.,1.,1.])
        elif self.stable == "min":
            self.w = np.ones(6)*1.5
            #self.zita = 0.707 * np.ones(6)
            self.zita = 0.1 * np.ones(6)
        elif self.stable == "mid":
            self.w = np.array([12,12,12,12,12,12])
            self.zita = np.array([0.5,0.5,0.5,0.5,0.5,0.5])
        elif self.stable == "rand":
            w = random.uniform(1.5, 20) 
            zita = random.uniform(0.1, 1) 
            self.w = np.array([w,w,w,w,w,w]) 
            self.zita = np.array([zita, zita, zita, zita, zita, zita])
        elif self.stable == "rand_static":
            if not self.param_change:
                w = random.uniform(1.5, 20) 
                zita = random.uniform(0.1, 1)
                self.w = np.array([w,w,w,w,w,w]) 
                self.zita = np.array([zita, zita, zita, zita, zita, zita])
                self.param_change = True
        else:
            w = np.nan_to_num(w)
            zita = np.nan_to_num(zita)
            self.w = np.clip(w, a_min=1.5, a_max=20.)
            self.zita = np.clip(zita, a_min=0.1, a_max=1.)

        # print(f"this episode w,sita is {self.w} , {self.zita}")
        # zita = np.array([0.2,0.2,0.2,1.,1.,1.])
        self.k1 = np.eye(n)
        self.k2 = 2 * self.zita * self.w * np.eye(n) #阻尼矩阵 （）
        self.k3 = self.w**2 * np.eye(n) #刚度矩阵



    #y_n[:3] is pos_error ,y_n[3:]is angle error
    #need vel error ,omega error
    # def update_state():


def admittance_expect_state():
    return 0
class Admittance_Euler():
    def __init__(self,):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.F = F

    def euler(self,):
        return self.k1



def main():
    n = 6
    w = np.array([5,5,5,5,5,5])
    zita = np.array([0.05,0.05,0.05,0.05,0.05,0.05])
    k1 = np.eye(n)
        # k2 = w**2 * np.eye(n) 
        # k3 = 2 * zita * w * np.eye(n)
    k2 = 2 * zita * w * np.eye(n)
    k3 = w**2 * np.eye(n) 
        # h = 0.1
    h = 0.1 * np.ones(n).reshape(n,1)
    admit = Admittance_rk(k1 = k1,
                            k2 = k2,
                            k3 = k3,
                            h = h)
    stable = "net"
    #admit = Admittance_rk(F=F,stable=stable)
    ys = []
    vels = []
    accs = []
    pa = np.array([-0.4489414, 0.97179586 , 1.4593623   ,0.7348407,  -0.5300652,  -0.15082535,
                    0.04522247 , 0.19175123 ,-0.15200493 , 0.07074573, -0.4125189,  -1.48482])
    # w = pa[:6]
    # zita = pa[6:]
    admit.update_parameter( w, zita, n=6)
    #print(admit.k2, admit.k3)
    for i in range(10):
        admit.F = np.array([0.,14.,0.,0.,0.,0.]).reshape(6,1)#50 * np.cos(i) * np.ones(n).reshape(n,1)
        y,vel,acc = admit.control_rk_2()
        y = y.reshape(1,n)[0]
        vel = vel.reshape(1,n)[0]
        acc = acc.reshape(1,n)[0]
        print(y,vel,acc)
        ys.append(y[0])
        vels.append(vel[0])
        accs.append(acc[0])
    plt.plot(range(len(ys)),ys)
    plt.show()

if __name__ == "__main__":
    main()