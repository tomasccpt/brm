import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation

class Sunfounder():
    def __init__(self, teta0=0, teta1=0, teta2=0, teta3=0):
        
        self.teta0 = teta0/180*np.pi
        self.teta1 = teta1/180*np.pi
        self.teta2 = teta2/180*np.pi
        self.teta3 = teta3/180*np.pi
        
        self.l1 = 5
        self.l2 = 3
        self.l3 = 2
        
        self.w1 = 0.2
        self.w2 = 0.2
    
        self.base = [[1,0,0],[0.707,0.707,0], [0,1,0],[-0.707,0.707,0], [-1,0,0],[-0.707,-0.707,0], [0,-1,0], [0.707,-0.707,0], 
                     [3, 0, -2], [2.121, 2.121, -2], [-2.121, 2.121, -2], [-3, 0, -2], [0, 3, -2], [-2.121, -2.121, -2], [0, -3, -2], [2.121, -2.121, -2],
                     [4, 0, -4], [2.828, 2.828, -4], [-2.828, 2.828, -4], [-4, 0, -4], [0, 4, -4], [-2.828, -2.828, -4], [0, -4, -4], [2.828, -2.828, -4]                     
                    ]
        
        self.colors = (*([(120/255,120/255,120/255)]*len(self.base)), *([(30/255,50/255,180/255)]*10), *([(15/255,240/255,60/255)]*10), *([(15/255,0/255,0/255)]*10))
        self.colors = np.array(self.colors)
    
    def set_angle(self, teta0=0, teta1=0, teta2=0, teta3=0):        
        self.teta0 = teta0/180*np.pi
        self.teta1 = teta1/180*np.pi
        self.teta2 = teta2/180*np.pi
        self.teta3 = teta3/180*np.pi

    def set_voltages(self, V0=0, V1=0, V2=0, V3=0):

        V0 = min(max(V0, 0), 255)
        V1 = min(max(V1, 0), 255)
        V2 = min(max(V2, 0), 255)
        V3 = min(max(V3, 0), 255)

        self.teta0 = (V0/255 + 0.5)*np.pi
        self.teta1 = (V1/255)*np.pi
        self.teta2 = (V2/255)*np.pi
        self.teta3 = (V3/255)*np.pi


    def draw(self):
        points = self.base.copy()
        width_vec =  np.array([np.sin(self.teta0),np.cos(self.teta0),0])
        claw_factor = np.array([np.sin(self.teta3/2)*np.sin(self.teta0),np.sin(self.teta3/2)*np.cos(self.teta0),0]) ##its more complicated than this. this is wrong
        
        alfa = self.l1 * np.cos(self.teta1)
        beta = alfa + self.l2*np.cos(self.teta1+self.teta2 - np.pi)
        beta3 = alfa + (self.l2+self.l3)*np.cos(self.teta1+self.teta2 - np.pi)
        
        p1 = np.array([np.cos(self.teta0)*alfa, np.sin(self.teta0)*alfa, self.l1*np.sin(self.teta1)])
        p2 = np.array([np.cos(self.teta0)*beta,np.sin(self.teta0)*beta, self.l1 * np.sin(self.teta1) + self.l2 * np.sin(self.teta2 + self.teta1 - np.pi)])
        
        p3 = np.array([np.cos(self.teta0)*beta3,np.sin(self.teta0)*beta3, (self.l2+self.l3) * np.sin(self.teta1) + (self.l2+self.l3) * np.sin(self.teta2 + self.teta1 - np.pi)])
        
        
        for i in range(0,5):
            points.append(i*p1/5+self.w1/2*width_vec)
            points.append(i*p1/5-self.w1/2*width_vec)

        for i in range(0,5):
            points.append(p1+i*(p2-p1)/5+self.w2/2*width_vec)
            points.append(p1+i*(p2-p1)/5-self.w2/2*width_vec)

            
        for i in range(0,5):
            points.append(p2+i*((p3-p2)*np.cos(self.teta3/2)+claw_factor)/5+self.w2/2*width_vec)
            points.append(p2+i*((p3-p2)*np.cos(self.teta3/2)-claw_factor)/5-self.w2/2*width_vec)
        
        points = np.array(points)

        return points, self.colors
    


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def update_arm_plot(robot_arm, voltages = [0,0,0,0], random_v=True):
    if random_v:
        voltages = np.random.randint(0, 256, size=4)

    robot_arm.set_voltages(*voltages)
    points, colors = robot_arm.draw()

    
    ax.clear()
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, c=colors)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-2, 8)

    return sc,


if __name__ == "__main__":
    robot_arm = Sunfounder()
    ani = animation.FuncAnimation(fig, update_arm_plot, frames=100, interval=200)
    plt.show()


     