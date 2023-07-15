import numpy as np
import cv2
from zmqRemoteApi import RemoteAPIClient
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox

class Simulation_App():
    def __init__(self):
        self.window = Tk()
        self.window.title('БПЛА')
        self.window.geometry('570x300')

        self.end = 0

        self.x_point = []
        self.y_point = []
        self.h = 1
        self.v = 0.5

        self.to_point_flag = False
        self.forward_flag = False
        self.spline_flag = False
        self.search_flag = False

        self.x_lb = Label(text="Point X: ", font="device 14 normal roman")
        self.x_lb.place(x=60, y=40)

        self.y_lb = Label(text="Point Y: ", font="device 14 normal roman")
        self.y_lb.place(x=60, y=80)

        self.height_lb = Label(text="Height: ", font="device 14 normal roman")
        self.height_lb.place(x=60, y=120)

        self.speed_lb = Label(text="Speed: ", font="device 14 normal roman")
        self.speed_lb.place(x=60, y=160)

        self.x_tf = Entry()
        self.x_tf.place(x=160, y=45)

        self.y_tf = Entry()
        self.y_tf.place(x=160, y=85)

        self.height_tf = Entry()
        self.height_tf.place(x=160, y=125)

        self.speed_tf = Entry()
        self.speed_tf.place(x=160, y=165)

        self.to_point = IntVar()
        self.mission_1_to_point = Checkbutton(text="Mission 1: to point", font="device 14 normal roman",
                                              variable=self.to_point, command=self.select)
        self.mission_1_to_point.place(x=320, y=35)

        self.uniform = IntVar()
        self.mission_2_uniform = Checkbutton(text="Mission 2: forward", font="device 14 normal roman",
                                             variable=self.uniform, command=self.select)
        self.mission_2_uniform.place(x=320, y=75)

        self.spline = IntVar()
        self.mission_3_spline = Checkbutton(text="Mission 3: spline", font="device 14 normal roman",
                                            variable=self.spline, command=self.select)
        self.mission_3_spline.place(x=320, y=115)

        self.search = IntVar()
        self.mission_4_search = Checkbutton(text="Mission 4: search", font="device 14 normal roman",
                                            variable=self.search, command=self.select)
        self.mission_4_search.place(x=320, y=155)

        self.start_btn = Button(text='Start', font="device 19 normal roman", command=self.start)
        self.start_btn.place(x=330, y=220)

        self.stop_btn = Button(text='Stop', font="device 19 normal roman", command=self.stop)
        self.stop_btn.place(x=430, y=220)

    def start(self):
        sim.startSimulation()

    def parameters(self):
        self.x_point = (self.x_tf.get())
        self.y_point = (self.y_tf.get())
        self.h = (self.height_tf.get())
        self.v = (self.speed_tf.get())

        self.x_point = list(map(float, self.x_point.split()))
        self.y_point = list(map(float, self.y_point.split()))
        self.h = list(map(float, self.h.split()))
        self.v = list(map(float, self.v.split()))
        return(self.x_point, self.y_point, self.h, self.v)


    def select(self):
        if self.to_point.get() == 1:
            self.to_point_flag = True
            self.forward_flag = False
            self.spline_flag = False
            self.search_flag = False

        if self.uniform.get() == 1:
            self.to_point_flag = False
            self.forward_flag = True
            self.spline_flag = False
            self.search_flag = False

        if self.spline.get() == 1:
            self.to_point_flag = False
            self.forward_flag = False
            self.spline_flag = True
            self.search_flag = False

        if self.search.get() == 1:
            self.to_point_flag = False
            self.forward_flag = False
            self.spline_flag = False
            self.search_flag = True

    def flags(self):
        return self.to_point_flag, self.forward_flag, self.spline_flag, self.search_flag


    def stop(self):
        sim.stopSimulation()
        self.end = 1

    def controll(self):
        self.window.update()

    def show_cubes(self, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10):
        self.msg1 = 'found cubes coordinates\n'
        self.msg2 = f'1 : {round(x1, 4), round(y1, 4)}\n'
        self.msg3 = f'2 : {round(x2, 4), round(y2, 4)}\n'
        self.msg4 = f'3 : {round(x3, 4), round(y3, 4)}\n'
        self.msg5 = f'4 : {round(x4, 4), round(y4, 4)}\n'
        self.msg6 = f'5 : {round(x5, 4), round(y5, 4)}\n'
        self.msg7 = f'6 : {round(x6, 4), round(y6, 4)}\n'
        self.msg8 = f'7 : {round(x7, 4), round(y7, 4)}\n'
        self.msg9 = f'8 : {round(x8, 4), round(y8, 4)}\n'
        self.msg10 = f'9 : {round(x9, 4), round(y9, 4)}\n'
        self.msg11 = f'10 : {round(x10, 4), round(y10, 4)}'
        self.msg = self.msg1 + self.msg2 + self.msg3 + self.msg4 + self.msg5 + self.msg6 + self.msg7 + self.msg8 + self.msg9 + self.msg10 + self.msg11
        top = Toplevel()
        top.geometry('300x350')
        Label(top, text = self.msg, font="device 14 normal roman").pack(expand=1)
        top.mainloop()

    def finish_mission(self):
        self.msg = 'Finish mission'
        messagebox.showinfo(title="END", message=self.msg)


class Trajectory():
    def __init__(self):
        self.way_x = np.zeros(int(Time / step))
        self.way_y = np.zeros(int(Time / step))
        self.way_z = np.zeros(int(Time / step))
        self.Vo = 1
        self.speed_boost = 0.005
        self.speed_less = 0.005
        self.way_s_done = 0
        self.way_x_done = 0
        self.way_y_done = 0

        self.angle_spiral = 0
        self.r = 0
        self.veloc_w = 0.01

    def forward_straight(self, i, X, Y, Z):
        self.way_x[i] = X
        self.way_y[i] = Y
        self.way_z[i] = Z
        return self.way_x[i], self.way_y[i], self.way_z[i]

    def detour_start(self, X, Y, V):

        if X > 0 and Y > 0:
            self.Angle = np.arctan2(Y, X)
        if X > 0 and Y < 0:
            self.Angle = np.arctan2(Y, X)
        if X < 0 and Y >= 0:
            self.Angle = np.arctan2(Y, X) - np.pi
        if X < 0 and Y < 0:
            self.Angle = np.arctan2(Y, X) + np.pi
        if X == 0 and Y > 0:
            self.Angle = np.pi / 2
        if X == 0 and Y < 0:
            self.Angle = -np.pi / 2
        if X == 0 and Y == 0:
            self.Angle = 0

        self.Vx_new = np.cos(self.Angle) * V
        self.Vy_new = np.sin(self.Angle) * V

        if X >= 0:
            self.way_x_detour = self.Vx_new
            self.way_y_detour = self.Vy_new
        if X < 0:
            self.way_x_detour = -self.Vx_new
            self.way_y_detour = -self.Vy_new

        if abs(self.way_x_detour) > abs(X):
            self.way_x_detour = X
        if abs(self.way_y_detour) > abs(Y):
            self.way_y_detour = Y

        return self.way_x_detour, self.way_y_detour

    def uniform_motion(self, i, X, Y, Z, V, start, cop_x, cop_y):
        if start == 0:
            self.way_x[i - 1] = cop_x
            self.way_y[i - 1] = cop_y
            self.ang_x = X - self.way_x[i - 1]
            self.ang_y = Y - self.way_y[i - 1]
            self.way_s = np.sqrt((X - self.way_x[i-1]) ** 2 + (Y - self.way_y[i-1]) ** 2)

            if self.ang_x > 0 and self.ang_y > 0:
                self.angle = np.arctan2(self.ang_y, self.ang_x)
            if self.ang_x > 0 and self.ang_y < 0:
                self.angle = np.arctan2(self.ang_y, self.ang_x)
            if self.ang_x < 0 and self.ang_y >= 0:
                self.angle = np.arctan2(self.ang_y, self.ang_x) - np.pi
            if self.ang_x < 0 and self.ang_y < 0:
                self.angle = np.arctan2(self.ang_y, self.ang_x) + np.pi
            if self.ang_x == 0 and self.ang_y > 0:
                self.angle = np.pi / 2
            if self.ang_x == 0 and self.ang_y < 0:
                self.angle = -np.pi / 2
            if self.ang_x == 0 and self.ang_y == 0:
                self.angle = 0
        if V > 1:
            self.Vo = self.Vo + self.speed_boost
            self.speed_boost = self.speed_boost + 0.001
            if self.Vo >= V:
                self.Vo = V
                self.speed_boost = 0
        else:
            self.Vo = V
        if self.way_s_done >= 0.85 * self.way_s:
            self.Vo = self.Vo - self.speed_less
            self.speed_less = self.speed_less + 0.001
            if self.Vo < 0.1:
                self.Vo = 0.1
        else:
            self.speed_less = 0.001

        self.Vx = np.cos(self.angle) * self.Vo
        self.Vy = np.sin(self.angle) * self.Vo
        if self.ang_x >= 0:
            if X <= self.way_x[i - 1] + self.Vx * step:
                self.way_x[i] = X
            else:
                self.way_x[i] = self.way_x[i - 1] + self.Vx * step
            self.way_y[i] = self.way_y[i - 1] + self.Vy * step
            if Y <= self.way_y[i - 1] + self.Vy * step and self.ang_y > 0:
                self.way_y[i] = Y
            if Y >= self.way_y[i - 1] + self.Vy * step and self.ang_y < 0:
                self.way_y[i] = Y
        if self.ang_x < 0:
            if X >= self.way_x[i - 1] - self.Vx * step:
                self.way_x[i] = X
            else:
                self.way_x[i] = self.way_x[i - 1] - self.Vx * step
            self.way_y[i] = self.way_y[i - 1] - self.Vy * step
            if Y <= self.way_y[i - 1] - self.Vy * step and self.ang_y > 0:
                self.way_y[i] = Y
            if Y >= self.way_y[i - 1] - self.Vy * step and self.ang_y < 0:
                self.way_y[i] = Y
        self.way_s_done = self.way_s_done + self.Vo * step
        self.way_z[i] = Z
        return self.way_x[i], self.way_y[i], self.way_z[i]

    def speed_cubic_spline(self, s, x, y, v):
        self.sigment_want = v * step
        self.n = len(x)
        self.h = np.zeros(self.n - 1)
        for i in range(self.n - 1):
            self.h[i] = x[i + 1] - x[i]

        self.A = np.zeros((self.n, self.n))
        self.A[0][0] = 1
        self.A[self.n - 1][self.n - 1] = 1
        for i in range(1, self.n - 1):
            self.A[i][i] = 2 * (self.h[i - 1] + self.h[i])
            self.A[i][i - 1] = self.h[i - 1]
            self.A[i][i + 1] = self.h[i]

        self.b = np.zeros(self.n)
        for i in range(1, self.n - 1):
            self.b[i] = 3 * ((y[i + 1] - y[i]) / self.h[i] - (y[i] - y[i - 1]) / self.h[i - 1])

        self.c = np.linalg.solve(self.A, self.b)
        self.d = np.zeros(self.n - 1)
        self.b = np.zeros(self.n - 1)
        for i in range(self.n - 1):
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3 * self.h[i])
            self.b[i] = (y[i + 1] - y[i]) / self.h[i] - (self.h[i] / 3) * (2 * self.c[i] + self.c[i + 1])
        self.n = len(x)
        self.xx = np.linspace(x[0], x[-1], 10000)
        self.yy = np.zeros(len(self.xx))
        self.l = 0
        self.spline_x = []
        self.spline_y = []
        for i in range(len(self.xx)):
            j = 0
            while j < self.n - 1 and self.xx[i] > x[j + 1]:
                j += 1
            self.yy[i] = y[j] + self.b[j] * (self.xx[i] - x[j]) + self.c[j] * (self.xx[i] - x[j]) ** 2 + self.d[j] * (self.xx[i] - x[j]) ** 3

        self.spline_x.append(self.xx[0])
        self.spline_y.append(self.yy[0])
        for k in range(len(self.xx)):
            self.len_sigment = np.sqrt((self.xx[self.l] - self.xx[k]) ** 2 + (self.yy[self.l] - self.yy[k]) ** 2)
            if self.sigment_want <= self.len_sigment:
                self.spline_x.append(self.xx[k])
                self.spline_y.append(self.yy[k])
                self.l = k
        self.spline_x.append(self.xx[len(self.xx) - 1])
        self.spline_y.append(self.yy[len(self.xx) - 1])
        if s >= len(self.spline_x) - 1:
            return self.xx[len(self.xx) - 1], self.yy[len(self.yy) - 1]
        else:
            return self.spline_x[s], self.spline_y[s]

    def spiral(self, v):
        self.v_start = self.veloc_w * self.r / 0.05
        self.dist_vit = 8
        if self.v_start >= v:
            self.veloc_w = v * 0.05 / self.r
        self.r = self.dist_vit / (2 * np.pi) * self.angle_spiral
        self.angle_spiral = self.angle_spiral + self.veloc_w
        x = self.r * np.cos(self.angle_spiral)
        y = self.r * np.sin(self.angle_spiral)
        return x, y


class Air_Robot():
    def __init__(self, mass_dvig, mass_base, blade, l, Fd, Md):
        self.m = mass_dvig
        self.M = mass_base
        self.mass = 4 * self.m + self.M
        self.blade = blade
        self.l = l
        self.thrust_coefficient = Fd
        self.moment_coefficient = Md

        self.g = 9.81
        self.F_takeoff = self.mass * self.g

        self.wb = 0
        self.wr = 0
        self.wf = 0
        self.wl = 0

        self.ang_course = np.zeros(int(Time / step))

    def get_acceleration(self):
        self.acc = np.zeros(3)
        self.acc = sim.getStringSignal("ACC")
        if self.acc != None:
            self.acc = sim.unpackFloatTable(self.acc)
            return self.acc[0], self.acc[1], self.acc[2]
        else:
            return 0, 0, 0

    def get_angular_vel(self):
        self.angels = np.zeros(3)
        self.angels = sim.getStringSignal("GYRO")
        if self.angels != None:
            self.angels = sim.unpackFloatTable(self.angels)
            return round(self.angels[0], 4), round(self.angels[1], 4), round(self.angels[2], 4)
        else:
            return 0, 0, 0

    def get_pos(self):
        self.position = np.zeros(3)
        self.position = sim.getStringSignal("GPS")
        if self.position != None:
            self.position = sim.unpackFloatTable(self.position)

            self.hight = sim.handleProximitySensor(dist_laser)
            self.position[2] = self.hight[1]
            return round(self.position[0], 4), round(self.position[1], 4), round(self.position[2], 4)
        else:
            return 0, 0, 0

    def barriers(self, bar_sen_l, bar_sen_f, bar_sen_r, s_l, s_r):
        self.bar_l = sim.handleProximitySensor(bar_sen_l)
        self.bar_f = sim.handleProximitySensor(bar_sen_f)
        self.bar_r = sim.handleProximitySensor(bar_sen_r)
        self.bar_Sl = sim.handleProximitySensor(s_l)
        self.bar_Sr = sim.handleProximitySensor(s_r)
        return self.bar_l[0], self.bar_f[0], self.bar_r[0], self.bar_l[1], self.bar_f[1], self.bar_r[1], self.bar_Sl[0], self.bar_Sr[0], self.bar_Sl[1], self.bar_Sr[1]

    def horizontal_controll(self, base, x_wish, y_wish, x_gps, y_gps, psi):
        self.base = base
        self.transion_matrix = sim.getObjectMatrix(self.base, sim.handle_world)
        self.Ox = [1, 0, 0]
        self.Ox = sim.multiplyVector(self.transion_matrix, self.Ox)
        self.Oy = [0, 1, 0]
        self.Oy = sim.multiplyVector(self.transion_matrix, self.Oy)

        self.x_copter = x_gps * np.cos(psi) + y_gps * np.sin(psi)
        self.y_copter = -x_gps * np.sin(psi) + y_gps * np.cos(psi)

        self.x_wish_c = x_wish * np.cos(psi) + y_wish * np.sin(psi) - self.x_copter
        self.y_wish_c = -x_wish * np.sin(psi) + y_wish * np.cos(psi) - self.y_copter

        self.x_copter = 0
        self.y_copter = 0

        self.err_teta = self.Ox[2] - self.transion_matrix[11]
        self.err_fi = self.Oy[2] - self.transion_matrix[11]

        return self.err_teta, self.err_fi, self.x_wish_c, self.y_wish_c, self.x_copter, self.y_copter

    def rotation_controll(self, copter_x, copter_y, x, y, psi):

        self.ex = (x - copter_x)
        self.ey = (y - copter_y)
        self.rx = np.cos(psi+np.pi/4)
        self.ry = np.sin(psi+np.pi/4)

        self.ang_course[i] = np.arccos(((self.ex * self.rx) + (self.ey * self.ry)) / (np.sqrt(self.ex**2 +self.ey ** 2) * np.sqrt(self.rx**2 + self.ry**2)))
        self.ang_course[i] = self.ang_course[i] * np.sign(-self.ex * np.sin(psi+np.pi/4) + self.ey * np.cos(psi+np.pi/4))
        if abs(self.ex) <= 0.0001 and abs(self.ey) <= 0.0001:
            self.ang_course[i] = 0
        return self.ang_course[i]


    def velocities_calculator(self, u1, u2, u3, u4):
        u1 = u1 + self.F_takeoff
        self.wb_wish = np.sqrt(abs(1 / (4 * self.thrust_coefficient) * u1 - 1 / (2 * self.thrust_coefficient * self.l) * u3 + 1 / (4 * self.moment_coefficient) * u4))
        self.wr_wish = np.sqrt(abs(1 / (4 * self.thrust_coefficient) * u1 - 1 / (2 * self.thrust_coefficient * self.l) * u2 - 1 / (4 * self.moment_coefficient) * u4))
        self.wf_wish = np.sqrt(abs(1 / (4 * self.thrust_coefficient) * u1 + 1 / (2 * self.thrust_coefficient * self.l) * u3 + 1 / (4 * self.moment_coefficient) * u4))
        self.wl_wish = np.sqrt(abs(1 / (4 * self.thrust_coefficient) * u1 + 1 / (2 * self.thrust_coefficient * self.l) * u2 - 1 / (4 * self.moment_coefficient) * u4))
        return self.wb_wish, self.wr_wish, self.wf_wish, self.wl_wish

    def move(self, motor1, motor2, motor3, motor4, w1, w2, w3, w4):  # back, right, front, left
        self.Td = 0.05
        self.wb = self.wb + (w1 - self.wb) * step / self.Td
        self.wr = self.wr + (w2 - self.wr) * step / self.Td
        self.wf = self.wf + (w3 - self.wf) * step / self.Td
        self.wl = self.wl + (w4 - self.wl) * step / self.Td

        self.Fb = self.wb ** 2 * self.thrust_coefficient
        self.Fr = self.wr ** 2 * self.thrust_coefficient
        self.Ff = self.wf ** 2 * self.thrust_coefficient
        self.Fl = self.wl ** 2 * self.thrust_coefficient

        self.Mb = self.wb ** 2 * self.moment_coefficient
        self.Mr = self.wr ** 2 * self.moment_coefficient
        self.Mf = self.wf ** 2 * self.moment_coefficient
        self.Ml = self.wl ** 2 * self.moment_coefficient

        sim.addForce(motor1, (0, 0, 0), (0, 0, self.Fb))  # куда, позиция, сколько силы
        sim.addForce(motor3, (0, 0, 0), (0, 0, self.Ff))
        sim.addForce(motor2, (0, 0, 0), (0, 0, self.Fr))
        sim.addForce(motor4, (0, 0, 0), (0, 0, self.Fl))

        sim.addForce(motor4, (0, -self.blade, 0), (-self.Ml, 0, 0))
        sim.addForce(motor4, (0, self.blade, 0), (self.Ml, 0, 0))

        sim.addForce(motor1, (0, self.blade, 0), (-self.Mb, 0, 0))  # пара моментов
        sim.addForce(motor1, (0, -self.blade, 0), (self.Mb, 0, 0))

        sim.addForce(motor3, (0, -self.blade, 0), (self.Mf, 0, 0))
        sim.addForce(motor3, (0, self.blade, 0), (-self.Mf, 0, 0))

        sim.addForce(motor2, (0, self.blade, 0), (self.Mr, 0, 0))
        sim.addForce(motor2, (0, -self.blade, 0), (-self.Mr, 0, 0))


class Camera():
    def detect(self, cop_x, cop_y, cop_h):
        self.color_red = (0, 0, 0)
        self.color_blue = (255, 0, 0)
        self.img, self.resX, self.resY = sim.getVisionSensorCharImage(visionSensorHandle)
        self.img = np.frombuffer(self.img, dtype=np.uint8).reshape(self.resY, self.resX, 3)
        self.img = cv2.flip(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), 0)
        self.Centr_X = int(self.resX / 2)
        self.Centr_Y = int(self.resY / 2)
        self.h_min = np.array((0, 0, 161), np.uint8)
        self.h_max = np.array((33, 71, 255), np.uint8)
        self.thresh = cv2.inRange(self.img, self.h_min, self.h_max)
        self.moments = cv2.moments(self.thresh, 1)
        self.dM01 = self.moments['m01']
        self.dM10 = self.moments['m10']
        self.dArea = self.moments['m00']
        self.offset_x, self.offset_y = 0, 0
        self.offset_x_global, self.offset_y_global = 0, 0

        if self.dArea > 7:
            self.x_cam = int(self.dM10 / self.dArea)
            self.y_cam = int(self.dM01 / self.dArea)
            self.offset_x = (self.x_cam - self.Centr_X)
            self.offset_y = (self.Centr_Y - self.y_cam)

            self.offset_x_global = (self.offset_x * (cop_h-0.03) / 128) + abs(cop_x)
            self.offset_y_global = (self.offset_y * (cop_h-0.03) / 128) + abs(cop_y)
            self.offset_x_global = self.offset_x_global * np.sign(cop_x)
            self.offset_y_global = self.offset_y_global * np.sign(cop_y)

            cv2.circle(self.img, (self.x_cam, self.y_cam), 2, self.color_blue, 1)
            cv2.putText(self.img, "x%d;y%d" % (self.x_cam, self.y_cam), (self.x_cam + 5, self.y_cam - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color_blue, 1)

            cv2.circle(self.img, (self.Centr_X, self.Centr_Y), 2, self.color_red, 1)
            cv2.putText(self.img, "x%d;y%d" % (self.x_cam - self.Centr_X, self.Centr_Y - self.y_cam),
                        (self.Centr_X + 5, self.Centr_Y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color_red, 1)
            cv2.line(self.img, (self.x_cam, self.y_cam), (self.Centr_X, self.Centr_Y), self.color_blue, 1)

        cv2.imshow('CopterIsSeeing', self.img)  # название окна и вывод изображения
        cv2.imshow('result', self.thresh)
        cv2.waitKey(1)  # число-миллисекунд через сколько обновится изображение 0-ждет любую клавишу для отображения

        return self.offset_x, self.offset_y, self.offset_x_global, self.offset_y_global

    def translation_close(self):
        cv2.destroyAllWindows()


class Calman_Filrter():
    def __init__(self):
        self.Dinamic = ([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
        self.H = np.zeros((9, 15))
        self.H[3][6] = 1
        self.H[4][7] = 1
        self.H[5][8] = 1
        self.H[6][9] = 1
        self.H[7][10] = 1
        self.H[8][11] = 1
        self.P = np.identity(15) * 10
        self.g = 9.81
        self.trash = 0.1
        self.R = np.identity(9) * self.trash
        self.model_ups_acc = 1
        self.model_ups_gy = 70
        self.Q = np.diag([self.model_ups_acc, self.model_ups_acc, self.model_ups_acc,
                          0, 0, 0,
                          0, 0, 0,
                          self.model_ups_gy, self.model_ups_gy, self.model_ups_gy,
                          0, 0, 0])
        self.Q = self.Q * step

    def solve_EFC(self, ax_input, ay_input, az_input, x_input, y_input, z_input, wx_input, wy_input, wz_input):
        self.input = (
            [[-ax_input], [-ay_input], [-az_input], [x_input], [y_input], [z_input], [wx_input], [wy_input],
             [wz_input]])

        self.fi = self.Dinamic[12][0]
        self.teta = self.Dinamic[13][0]
        self.psi = self.Dinamic[14][0]

        self.Rs = np.array([[np.cos(self.psi) * np.cos(self.teta),
                             np.cos(self.psi) * np.sin(self.fi) * np.sin(self.fi) - np.cos(self.fi) * np.sin(self.psi),
                             np.sin(self.fi) * np.sin(self.psi) + np.cos(self.psi) * np.sin(self.teta)],
                            [np.sin(self.psi) * np.cos(self.teta),
                             np.cos(self.fi) * np.cos(self.psi) + np.sin(self.teta) * np.sin(self.psi),
                             np.cos(self.fi) * np.sin(self.psi) * np.sin(self.teta) - np.cos(self.psi) * np.sin(
                                 self.fi)],
                            [-np.sin(self.teta), np.cos(self.teta) * np.sin(self.fi),
                             np.cos(self.fi) * np.cos(self.teta)]])

        self.Rw = np.array([[1, 0, -np.sin(self.teta)],
                            [0, np.cos(self.fi), np.sin(self.fi) * np.cos(self.teta)],
                            [0, -np.sin(self.fi), np.cos(self.fi) * np.cos(self.teta)]])

        self.F = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [self.Rs[0][0] * step, self.Rs[0][1] * step, self.Rs[0][2] * step, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0],
                           [self.Rs[1][0] * step, self.Rs[1][1] * step, self.Rs[1][2] * step, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0],
                           [self.Rs[2][0] * step, self.Rs[2][1] * step, self.Rs[2][2] * step, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0],
                           [self.Rs[0][0] * step ** 2 / 2, self.Rs[0][1] * step ** 2 / 2, self.Rs[0][2] * step ** 2 / 2,
                            step, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [self.Rs[1][0] * step ** 2 / 2, self.Rs[1][1] * step ** 2 / 2, self.Rs[1][2] * step ** 2 / 2,
                            0, step, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [self.Rs[2][0] * step ** 2 / 2, self.Rs[2][1] * step ** 2 / 2, self.Rs[2][2] * step ** 2 / 2,
                            0, 0, step, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, self.Rw[0][0] * step, self.Rw[0][1] * step, self.Rw[0][2] * step,
                            1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, self.Rw[1][0] * step, self.Rw[1][1] * step, self.Rw[1][2] * step,
                            0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, self.Rw[2][0] * step, self.Rw[2][1] * step, self.Rw[2][2] * step,
                            0, 0, 1]])

        self.H[0][0] = 1 + self.g * np.sin(self.teta)
        self.H[1][1] = 1 - self.g * np.cos(self.teta) * np.sin(self.fi)
        self.H[2][2] = 1 - self.g * np.cos(self.teta) * np.cos(self.fi)

        self.Dinamic = self.F.dot(self.Dinamic)
        self.P = (self.F.dot(self.P)).dot(self.F.transpose()) + self.Q
        self.K = (self.P.dot(self.H.transpose())).dot(
            np.linalg.inv(((self.H.dot(self.P)).dot(self.H.transpose())) + self.R))
        self.Dinamic = self.Dinamic + (self.K.dot((self.input - self.H.dot(self.Dinamic))))
        self.P = (np.identity(15) - self.K.dot(self.H)).dot(self.P)

        return self.Dinamic

    def reset_integrirovanie(self):
        self.Dinamic[14][0] = 0


class Regulator():
    def __init__(self, kp, ki, kd, vmin, umax):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.vmin = vmin
        self.umax = umax
        self.sig = np.zeros(int(Time / step))
        self.error = np.zeros(int(Time / step))
        self.error_sig = np.zeros(int(Time / step))
        self.ui_past = 0
        self.u = np.zeros(int(Time / step))

    def speed_limit(self, i, wish, have):
        self.err_v = wish - self.sig[i - 1]
        if abs(self.err_v) > self.vmin * step:
            self.sig[i] = self.sig[i - 1] + self.vmin * step * np.sign(self.err_v)
        else:
            self.sig[i] = self.sig[i - 1] + self.err_v
        self.error_sig[i] = self.sig[i] - have
        return self.error_sig[i]

    def pid(self, i, err, moment):
        self.error[i] = err + moment
        up = self.kp * self.error[i]
        # Ограничение интегральной составляющей
        if self.u[i - 1] < self.umax:
            ui = self.ui_past + self.ki * step * self.error[i]
        else:
            ui = self.ui_past
        ud = self.kd * (self.error[i] - self.error[i - 1]) / step
        self.u[i] = up + ui + ud
        self.ui_past = ui
        return self.u[i]


client = RemoteAPIClient()
sim = client.getObject('sim')
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)
rob_base = sim.getObject('/Copter')
motor_left = sim.getObject('/M_L')
motor_Back = sim.getObject('/M_B')
motor_Front = sim.getObject('/M_F')
motor_Right = sim.getObject('/M_R')
dist_laser = sim.getObject('/laser')
barrier_sensor_left = sim.getObject('/pr_sensor_l')
barrier_sensor_front = sim.getObject('/pr_sensor_f')
barrier_sensor_right = sim.getObject('/pr_sensor_r')
stop_left = sim.getObject('/pr_l_stop')
stop_right = sim.getObject('/pr_r_stop')
visionSensorHandle = sim.getObject('/Vision_sensor')

Time, step = 100000, 0.05
App = Simulation_App()
Copter = Air_Robot(0.5, 0.2, 0.05, 0.091, 7.5 * 10 ** -4, 1.2 * 10 ** -5)
Route = Trajectory()
Data = Calman_Filrter()
Cam = Camera()

U1 = Regulator(0.66, 0, 4, 10, 0)
Ang_fi = Regulator(0.193, 0, 0.115, 0, 0)
Ang_teta = Regulator(0.193, 0, 0.115, 0, 0)
Mx_y = Regulator(0.031, 0, 0.101, 20, 0)
My_x = Regulator(0.031, 0, 0.101, 20, 0)
U4 = Regulator(0.92, 0, 1.5, 0, 0)

start_move = False
start_rotation = False
Spline_Done = False
mission_1_to_point = False
mission_2_forward = False
mission_3_spline = False
mission_4_saerch = False
finish = False
get_param = False
x_points = []
y_points = []
flight_height = []
flight_speed = []

i = 0
j = 0
do_serch = -1
count_spline = -1
stab_rotation = 0
down = 0
end = 0
x_for_spline = []
y_for_spline = []
x_for_spline.append(0)
y_for_spline.append(0)

b_l, b_f, b_r = 0, 0, 0
u, v, w, x, y, z, p, q, r = 0, 0, 0, 0, 0, 0, 0, 0, 0
Mx, My = 0, 0
speed_limit_Mx = 0
speed_limit_My = 0
speed_limit_rotation = 0.07
err_teta_x, err_fi_y = 0, 0
v_coeff = 1
v_for_dif_mission = 1
div = 0
u_1 = np.zeros(int(Time / step))
u_2 = np.zeros(int(Time / step))
u_3 = np.zeros(int(Time / step))
u_4 = np.zeros(int(Time / step))

pos_x = np.zeros(int(Time / step))
pos_y = np.zeros(int(Time / step))
pos_z = np.zeros(int(Time / step))
fi = np.zeros(int(Time / step))
teta = np.zeros(int(Time / step))
rob_x = np.zeros(int(Time / step))
rob_y = np.zeros(int(Time / step))
course = np.zeros(int(Time / step))
psi = np.zeros(int(Time / step))
veloc_xy = np.zeros(int(Time / step))

trajectory_x = np.zeros(int(Time / step))
trajectory_y = np.zeros(int(Time / step))
trajectory_z = np.zeros(int(Time / step))
start_now = 0

cubes = np.zeros((10, 2))
count_cube = 0
found_cube = False
repeat = False
curse_reset = False

timer = np.zeros(int(Time / step))
pos_x_graph = []
pos_y_graph = []
tr_x_graph = []
tr_y_graph = []
speed_graph = []
h_graph = []
course_graph = []
time_graph = []

client.setStepping(True)
print("start")
state = sim.getSimulationState()

while state == 0:
    App.controll()
    state = sim.getSimulationState()

while state != 0:
    if get_param == False:
        x_points, y_points, flight_height, flight_speed = App.parameters()
        mission_1_to_point, mission_2_forward, mission_3_spline, mission_4_saerch = App.flags()
        get_param = True
    App.controll()

    x, y, z = Copter.get_pos()
    u, v, w = Copter.get_acceleration()
    p, q, r = Copter.get_angular_vel()
    result = Data.solve_EFC(u, v, w, x, y, z, p, q, r)
    pos_x[i], pos_y[i], pos_z[i] = x, y, z
    fi[i], teta[i], psi[i] = result[12][0], result[13][0], result[14][0]
    veloc_xy[i] = np.sqrt(result[3][0] ** 2 + result[4][0] ** 2)
    KS = 0.1*flight_speed[0]
    K = 4

    if (start_move == True and mission_1_to_point == True):
        v_for_dif_mission = 10
        count_point = len(x_points) - len(x_points) + j
        trajectory_x[i], trajectory_y[i], trajectory_z[i] = Route.uniform_motion(i, x_points[count_point],y_points[count_point],flight_height[0],flight_speed[0] * v_coeff, start_now,pos_x[i], pos_y[i])
        start_now = 1
        course[i] = Copter.rotation_controll(pos_x[i - 1], pos_y[i - 1], pos_x[i], pos_y[i], psi[i])
        if round(pos_x[i - 1], 1) == x_points[count_point] and round(pos_y[i - 1], 1) == y_points[count_point] and len(x_points) - j > 1:
            j = j + 1
            start_now = 0
        if abs(x_points[len(x_points)-1] - pos_x[i]) <= 2:
            course[i] = Copter.rotation_controll(pos_x[i], pos_y[i],x_points[count_point] + 0.5 * x_points[count_point] * np.sign(x_points[count_point]),y_points[count_point] + 0.5 * y_points[count_point] * np.sign(y_points[count_point]), psi[i])
            if abs(x_points[count_point] - pos_x[i]) <= 0.5 and abs(
                    y_points[count_point] - pos_y[i]) <= 0.5 and finish == False and mission_4_saerch == False:
                finish = True
                if finish == True:
                    App.finish_mission()

    if (start_move == True and mission_2_forward == True):
        v_for_dif_mission = 10
        count_point = len(x_points) - len(x_points) + j
        trajectory_x[i], trajectory_y[i], trajectory_z[i] = Route.uniform_motion(i, x_points[count_point],y_points[count_point],flight_height[0], flight_speed[0]*v_coeff, start_now, pos_x[i], pos_y[i])
        start_now = 1
        course[i] = Copter.rotation_controll(pos_x[i-1], pos_y[i-1], pos_x[i], pos_y[i], psi[i])
        if round(pos_x[i - 1], 1) == x_points[count_point] and round(pos_y[i - 1], 1) == y_points[count_point] and len(x_points) - j > 1:
            j = j + 1
            start_now = 0
        if abs(x_points[len(x_points)-1] - pos_x[i]) <= 2:
            course[i] = Copter.rotation_controll(pos_x[i], pos_y[i], x_points[count_point] + 0.5*x_points[count_point]*np.sign(x_points[count_point]), y_points[count_point] + 0.5*y_points[count_point]*np.sign(y_points[count_point]), psi[i])
            if abs(x_points[count_point] - pos_x[i]) <= 0.5 and abs(y_points[count_point] - pos_y[i]) <= 0.5 and finish == False and mission_4_saerch == False:
                finish = True
                if finish == True:
                    App.finish_mission()

    if (start_move == True and mission_3_spline == True):
        v_for_dif_mission = 1
        if count_spline == -1:
            x_for_spline[0], y_for_spline[0] = pos_x[1], pos_y[1]
            for a in range(len(x_points)):
                x_for_spline.append(x_points[a])
                y_for_spline.append(y_points[a])
            count_spline = count_spline + 1
        else:
            trajectory_x[i], trajectory_y[i] = Route.speed_cubic_spline(count_spline, x_for_spline, y_for_spline,flight_speed[0]*v_coeff)
            course[i] = Copter.rotation_controll(pos_x[i-1], pos_y[i-1], pos_x[i], pos_y[i], psi[i])
            trajectory_z[i] = flight_height[0]
            count_spline = count_spline + 1

        if abs(x_for_spline[len(x_for_spline)-1] - pos_x[i]) <= 2:
            course[i] = Copter.rotation_controll(pos_x[i], pos_y[i],x_for_spline[len(x_for_spline)-1] + 0.2*x_for_spline[len(x_for_spline)-1]*np.sign(x_for_spline[len(x_for_spline)-1]), y_for_spline[len(y_for_spline)-1] + 0.2*y_for_spline[len(y_for_spline)-1]*np.sign(y_for_spline[len(y_for_spline)-1]), psi[i])
            if abs(x_for_spline[len(x_for_spline)-1] - pos_x[i]) <= 0.5 and abs(y_for_spline[len(y_for_spline)-1] - pos_y[i]) <= 0.5 and finish == False:
                finish = True
                if finish == True:
                    App.finish_mission()

    if (start_move == True and mission_4_saerch == True):
        v_for_dif_mission = 10000
        if do_serch < 17:
            trajectory_x[i], trajectory_y[i], trajectory_z[i] = Route.uniform_motion(i, 0.001, 0, flight_height[0], flight_speed[1], start_now, pos_x[i], pos_y[i] )
            start_now = 1
            course[i] = Copter.rotation_controll(pos_x[i-1], pos_y[i-1], pos_x[i], pos_y[i], psi[i])
            if abs(round(pos_x[i - 1], 1)) <= 0.5 and abs(round(pos_y[i - 1], 1)) <= 0.5:
                course[i] = course[i-1]
                if (abs(course[i]) <= 0.05):
                    do_serch = do_serch + 1
        else:
            if psi[i] >= 6.28:
                Data.reset_integrirovanie()
            test = 0
            repeat = False
            camX, camY, cube_x, cube_y = Cam.detect(pos_x[i], pos_y[i], pos_z[i])
            trajectory_x[i], trajectory_y[i] = Route.spiral(flight_speed[0]*v_coeff)
            course[i] = Copter.rotation_controll(pos_x[i-1], pos_y[i-1], pos_x[i], pos_y[i], psi[i])
            trajectory_z[i] = flight_height[0]
            do_serch = do_serch + 1

            if camY > 0:
                cubes[count_cube][0] = cube_x
                cubes[count_cube][1] = cube_y
                found_cube = True
            if camY < 0 and found_cube == True:
                print("ухожу")
                if count_cube >= 1:
                    for test in range (count_cube):
                        print(test)
                        if abs(abs(cubes[test][0]) - abs(cubes[count_cube][0])) < 9.5 and abs(abs(cubes[test][1]) - abs(cubes[count_cube][1])) < 9.5:
                            cubes[test][0] = (cubes[test][0] + cubes[count_cube][0]) / 2
                            cubes[test][1] = (cubes[test][1] + cubes[count_cube][1]) / 2
                            repeat = True
                            cubes[count_cube][0] = 0
                            cubes[count_cube][1] = 0
                            print("повтор")
                        else:
                            print ("не повтор")
                    if repeat == True:
                        count_cube = count_cube - 1
                    found_cube = False
                    count_cube = count_cube + 1
                if count_cube == 0:
                    count_cube = 1
                    found_cube = False
                print("нашел",count_cube)
                print(cubes)
            if count_cube == 10:
                App.finish_mission()
                count_cube = 11

    if (start_move == True):
        if trajectory_x[i] == 0 and trajectory_y[i] == 0 and trajectory_z[i] == 0:
            trajectory_x[i], trajectory_y[i], trajectory_z[i] = trajectory_x[i-1], trajectory_y[i-1], trajectory_z[i-1]
            course[i] = course[i-1]
        if pos_z[i-1] - pos_z[i] > 1.5:
            pos_z[i] = pos_z[i-1]

        err_teta_x, err_fi_y, x_way_c, y_way_c, x_c, y_c = Copter.horizontal_controll(rob_base, trajectory_x[i - 2], trajectory_y[i - 2], pos_x[i],pos_y[i], psi[i])
        b_l, b_f, b_r, b_l_dist, b_f_dist, b_r_dist, s_l, s_r, s_l_dist, s_r_dist = Copter.barriers(barrier_sensor_left, barrier_sensor_front,barrier_sensor_right,stop_left, stop_right)
        x_way_c, y_way_c = Route.detour_start(x_way_c, y_way_c, flight_speed[0])
        v_coeff = 1
        if mission_3_spline == False:
            if veloc_xy[i] > flight_speed[0] + 0.034*flight_speed[0]:
                v_coeff = 0.9
                x_way_c, y_way_c = Route.detour_start(x_way_c, y_way_c, 0.4*flight_speed[0])

        if s_r == 1 and b_f == 0:
            v_coeff = v_for_dif_mission
            x_way_c, y_way_c = Route.detour_start(-(KS * flight_speed[0])/s_r_dist, 4, 0.005*flight_speed[0])

        if s_l == 1 and b_f == 0:
            v_coeff = v_for_dif_mission
            x_way_c, y_way_c = Route.detour_start(4, -(KS * flight_speed[0]) / s_l_dist, 0.005*flight_speed[0])

        if b_f == 1:
            v_coeff = v_for_dif_mission
            if b_r == 1 and b_l == 0:
                x_way_c, y_way_c = Route.detour_start(-(K*flight_speed[0])/b_f_dist, 4, flight_speed[0])
            elif b_r == 0 and b_l == 1:
                x_way_c, y_way_c = Route.detour_start(4, -(K * flight_speed[0]) / b_f_dist, flight_speed[0])
            else:
                x_way_c, y_way_c = Route.detour_start(4, -(2*K * flight_speed[0]) / b_f_dist, flight_speed[0])

        Mx = Mx_y.pid(i, Mx_y.speed_limit(i, x_way_c, x_c), 0)
        My = My_x.pid(i, My_x.speed_limit(i, y_way_c, y_c), 0)


    if (start_move == False):
        trajectory_x[i], trajectory_y[i], trajectory_z[i] = Route.forward_straight(i, pos_x[1], pos_y[1], 1)
        course[i] = Copter.rotation_controll(pos_x[i], pos_y[i], x_points[0], y_points[0], psi[i])
        err_teta_x, err_fi_y, x_way_c, y_way_c, x_c, y_c = Copter.horizontal_controll(rob_base, trajectory_x[i], trajectory_y[i], pos_x[i],pos_y[i], psi[i])
        if i == 0:
            Mx = 0
            My = 0
        else:
            Mx = Mx_y.pid(i, Mx_y.speed_limit(i, x_way_c, x_c), 0)
            My = My_x.pid(i, My_x.speed_limit(i, y_way_c, y_c), 0)
            speed_limit_Mx = 0.0001
            speed_limit_My = 0.0001
        if pos_z[i] >= 0.9:
            start_rotation = True
            print("rot")

    if (start_rotation == True and start_move == False):
        course[i] = Copter.rotation_controll(pos_x[i], pos_y[i], x_points[0], y_points[0], psi[i])
        speed_limit_Mx = 0.005
        speed_limit_My = 0.005
        if (abs(course[i]) <= 0.02):
            stab_rotation = stab_rotation + 1
            if stab_rotation == 15:
                start_move = True
                print("move")

    u_1[i] = U1.pid(i, U1.speed_limit(i, trajectory_z[i], pos_z[i]), 0)
    u_2[i] = Ang_fi.pid(i, err_fi_y, My)  # My
    u_3[i] = Ang_teta.pid(i, err_teta_x, Mx)  # Mx
    u_4[i] = U4.pid(i, course[i], 0)

    if abs(u_2[i]) > speed_limit_Mx:
        u_2[i] = speed_limit_Mx * np.sign(u_2[i])

    if abs(u_3[i]) > speed_limit_My:
        u_3[i] = speed_limit_My * np.sign(u_3[i])

    if abs(u_4[i]) > speed_limit_rotation:
        u_4[i] = speed_limit_rotation * np.sign(u_4[i])

    w1, w2, w3, w4 = Copter.velocities_calculator(u_1[i], -u_2[i], -u_3[i], u_4[i])  # -u_2[i], -u_3[i], u_4[i]
    Copter.move(motor_Back, motor_Right, motor_Front, motor_left, w1, w2, w3, w4)

    if App.end == 1:
        if mission_4_saerch == True:
            App.show_cubes(cubes[0][0], cubes[0][1], cubes[1][0], cubes[1][1], cubes[2][0], cubes[2][1], cubes[3][0], cubes[3][1], cubes[4][0], cubes[4][1],
                           cubes[5][0], cubes[5][1], cubes[6][0], cubes[6][1], cubes[7][0], cubes[7][1], cubes[8][0], cubes[8][1], cubes[9][0], cubes[9][1])
        plt.figure(figsize=(14, 10))
        plt.suptitle("Motion characteristics")
        plt.subplot(2, 2, 1)
        plt.title("trajectiry and robot position")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.plot(pos_x_graph, pos_y_graph, color='green')
        plt.plot(tr_x_graph, tr_y_graph, 'r--')
        plt.subplot(2, 2, 2)
        plt.title("height")
        plt.xlabel("time")
        plt.ylabel("height")
        plt.plot(time_graph, h_graph, color='black')
        plt.subplot(2, 2, 3)
        plt.title("speed")
        plt.xlabel("time")
        plt.ylabel("speed")
        plt.plot(time_graph, speed_graph, color='black')
        plt.subplot(2, 2, 4)
        plt.title("course error")
        plt.xlabel("time")
        plt.ylabel("error")
        plt.plot(time_graph, course_graph, color='black', label='course_error')
        plt.show()

    if i>5:
        pos_x_graph.append(pos_x[i])
        pos_y_graph.append(pos_y[i])
        tr_x_graph.append(trajectory_x[i])
        tr_y_graph.append(trajectory_y[i])
        if veloc_xy[i] > flight_speed[0] and i < 100:
            speed_graph.append(0)
        else:
            speed_graph.append(veloc_xy[i])
        h_graph.append(pos_z[i])
        course_graph.append(course[i])
        time_graph.append(timer[i-1])

    timer[i] = i * step
    i = i + 1
    client.step()