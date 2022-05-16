import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x,y2=np.loadtxt('outcomes/simd.txt',delimiter=' ',unpack=True)
x,y3=np.loadtxt('outcomes/avx.txt',delimiter=' ',unpack=True)
x,y4=np.loadtxt('outcomes/openmp.txt',delimiter=' ',unpack=True)
x,y5=np.loadtxt('outcomes/simd_openmp.txt',delimiter=' ',unpack=True)
x,y6=np.loadtxt('outcomes/avx_openmp.txt',delimiter=' ',unpack=True)
plt.plot(x,y2,label='simd')
plt.plot(x,y3,label='avx')
plt.plot(x,y4,label='openmp')
plt.plot(x,y5,label='simd_openmp')
plt.plot(x,y6,label='avx_openmp')
plt.xlabel('test scale/n')
plt.ylabel('time/ms')
plt.title('performance_comp')
plt.legend()
plt.savefig('images/AVX_comp.png')