import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x,y1=np.loadtxt('outcomes/ordinary.txt',delimiter=' ',unpack=True)
x,y2=np.loadtxt('outcomes/simd.txt',delimiter=' ',unpack=True)
x,y3=np.loadtxt('outcomes/openmp.txt',delimiter=' ',unpack=True)
x,y4=np.loadtxt('outcomes/simd_openmp.txt',delimiter=' ',unpack=True)
plt.plot(x,y1,label='ordinary')
plt.plot(x,y2,label='simd')
plt.plot(x,y3,label='openmp')
plt.plot(x,y4,label='simd_openmp')
plt.xlabel('test scale/n')
plt.ylabel('time/ms')
plt.title('performance_comp')
plt.legend()
plt.savefig('images/SIMD_comp.png')