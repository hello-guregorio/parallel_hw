import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x,y1=np.loadtxt('outcomes/ordinary_thread.txt',delimiter=' ',unpack=True)
x,y2=np.loadtxt('outcomes/ordinary_thread2.txt',delimiter=' ',unpack=True)
x,y3=np.loadtxt('outcomes/ordinary_thread4.txt',delimiter=' ',unpack=True)
x,y4=np.loadtxt('outcomes/ordinary_thread6.txt',delimiter=' ',unpack=True)
plt.plot(x,y1,label='ordinary')
plt.plot(x,y2,label='2 threads')
plt.plot(x,y3,label='4 threads')
plt.plot(x,y4,label='6 threads')
plt.xlabel('test scale/n')
plt.ylabel('time/ms')
plt.title('performance_comp')
plt.legend()
plt.savefig('images/Thread_comp.png')