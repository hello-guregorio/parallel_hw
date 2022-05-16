import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x,y1=np.loadtxt('outcomes/ordinary_school.txt',delimiter=' ',unpack=True)
x,y2=np.loadtxt('outcomes/openmp_school.txt',delimiter=' ',unpack=True)
x,y3=np.loadtxt('outcomes/neon_school.txt',delimiter=' ',unpack=True)
x,y4=np.loadtxt('outcomes/neon_openmp_school.txt',delimiter=' ',unpack=True)
plt.plot(x,y1,label='ordinary_kunpeng')
plt.plot(x,y2,label='neon')
plt.plot(x,y3,label='openmp_kunpeng')
plt.plot(x,y4,label='neon_openmp_kunpeng')
plt.xlabel('test scale/n')
plt.ylabel('time/ms')
plt.title('performance_comp')
plt.legend()
plt.savefig('images/Neon_comp.png')