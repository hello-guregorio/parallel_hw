import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x,y1=np.loadtxt('outcomes/openmp.txt',delimiter=' ',unpack=True)
x,y2=np.loadtxt('outcomes/openmp_static.txt',delimiter=' ',unpack=True)
x,y3=np.loadtxt('outcomes/openmp_dynamic.txt',delimiter=' ',unpack=True)
x,y4=np.loadtxt('outcomes/openmp_guided.txt',delimiter=' ',unpack=True)
x,y5=np.loadtxt('outcomes/ordinary.txt',delimiter=' ',unpack=True)
plt.plot(x,y1,label='openmp')
plt.plot(x,y2,label='openmp_static')
plt.plot(x,y3,label='openmp_dynamic')
plt.plot(x,y4,label='openmp_guided')
plt.plot(x,y5,label='ordinary')
plt.xlabel('test scale/n')
plt.ylabel('time/ms')
plt.title('performance_comp')
plt.legend()
plt.savefig('images/schedule.png')