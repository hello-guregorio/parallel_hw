import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x,y1=np.loadtxt('outcomes/openmp_school.txt',delimiter=' ',unpack=True)
x,y3=np.loadtxt('outcomes/ordinary_school.txt',delimiter=' ',unpack=True)
y2=np.loadtxt('outcomes/pthread.txt',unpack=True)
plt.plot(x,y1,label='openmp')
plt.plot(x,y2,label='pthread')
plt.plot(x,y3,label='ordinary')
plt.xlabel('test scale/n')
plt.ylabel('time/ms')
plt.title('performance_comp')
plt.legend()
plt.savefig('images/Neon_comp.png')