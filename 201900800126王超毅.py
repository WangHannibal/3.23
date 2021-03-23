# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 22:58:20 2021

@author: WangChaoYI
"""
#我的电脑分层多了之后会变的很卡所以只能到这了
import numpy as np
import matplotlib.pyplot as plt

xdata = np.array([8., 3., 9., 7., 16., 05., 3., 10., 4., 6.])
x0=np.ones(10)

xdata = np.concatenate((xdata,x0), axis=0)
xdata=xdata.reshape(2,10)
#print(xdata)
ydata = np.array([30., 21., 35., 27., 42., 24., 10., 38., 22., 25.])
ydata=ydata.reshape(1,10)
nlr1=0.
theta=np.array([7.5,70.])
theta=theta.reshape(2,1)
lr1=0.
n=10
alpha=0.01

def tem(x,y):
    end=np.zeros(x.shape)
    for i in range (len(xdata[0])):
        end+=((x*xdata[0][i]+y-ydata[0][i])**2)
    end=end/200000
    return end

xb = np.linspace(-100,100,1000)
yw = np.linspace(-10,10,1000)
X,Y = np.meshgrid(xb,yw)
Z=tem(Y,X)
plt.contourf(X,Y,Z)
thec=plt.contour(X,Y,Z)
plt.clabel(thec)
for j in range(0,2000):
    nlr1=np.sum((np.matmul(xdata.T,theta)-ydata.T)**2)
    delta=np.matmul(xdata.T,theta)-ydata.T
    lr1=nlr1/(2*n)
    for i in range(0,theta.shape[0]):
        theta[i]=theta[i]-alpha*(1/n)*np.sum(xdata[i].reshape(n,1)*delta)
    print(lr1)
    plt.scatter(theta[1],theta[0],marker = '.',c = 'r') 
plt.show()



plt.scatter(xdata[0],np.matmul(xdata.T,theta),color='yellow')
plt.scatter(xdata[0],ydata,color='blue')
plt.plot(xdata[0],np.matmul(xdata.T,theta),'y')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 25)
plt.ylim(0,50)
plt.show()
plt.close()




