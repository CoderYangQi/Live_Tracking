from ctypes import *
import ctypes
#import numpy as np
import struct 
import time
import math


    
dll = CDLL(r'E:\yq\code\MovingInterpolation2D\GAS.dll')
print(dll)

dll.GA_StartDebugLog(1)

print('测试开始')
a=dll.GA_OpenByIP(b'192.168.0.2',b'192.168.0.1',0,0)
print('打开板卡GA_Open返回值:',a)


a=dll.GA_StartLog()
print('打开日志功能，平时不用可以关闭本段代码，返回值:',a)
"产生的日志文件夹名字为 RunTimeLog 可以在电脑里面搜索该文件夹。通常在Pyhon安装路径"

a=dll. GA_Reset()
print('复位板卡GA_Reset返回值:',a)

a=dll.GA_EncOff(1)
print('关闭轴1编码器:',a)

# a=dll.GA_ZeroPos(1,1)
# print('清零轴1零位，返回值:',a)

a=dll.GA_AxisOn(1)
print('使能轴1返回值:',a)

a=dll.GA_SetCrdPrmSingleEX(1,2,1,2,0,0,0,0,0,0,c_double(2000),c_double(5),0,1,0,0,0,0,0,0,0,0)
print('建立2维坐标系，返回值:',a)
dPrfPosx = c_double(0)
dPrfPosy = c_double(0)
speed = 5
a=dll.GA_InitLookAheadSingleEX(1,0,speed,speed,speed,50,400,400,2,2,2,2,2,5,5,5,5,5,1,1,1,1,1)
print('初始化前瞻，返回值:',a)

a = dll.GA_CrdStart(1,0)
print('启动坐标系运动,返回值:',a)

# xValue = 0
# yValue = 10000
# a=dll.GA_LnXY(1,xValue,yValue,c_double(20.5),c_double(0.9),0,0,2)
print('插入2维插补数据,X=50000脉冲，Y=50000脉冲,返回值:',a)
a = dll.GA_GetPrfPos(1, byref(dPrfPosx),1,0)
dValue = dPrfPosx
print('获取轴1脉冲位置，返回值：',a,'获取值：',dValue)
a = dll.GA_GetPrfPos(2, byref(dPrfPosy),1,0)
dValue = dPrfPosy
print('获取轴2脉冲位置，返回值：',a,'获取值：',dValue)
def move(xValue,yValue):
    

    print(f"执行插补运动: X={xValue}, Y={yValue}")
    a = dll.GA_LnXY(1, int(xValue), int(yValue), c_double(20.5), c_double(0.9), 0, 0, 2)
    # control the movement 
    a=dll.GA_CrdData(1,0,0)
def stop():
    a= dll.GA_CrdData(1,0,0)
    axis = 2
    lMask = (0x0001 << (axis - 1))  # 计算掩码值
    lOption = 0  # 根据函数要求设置此参数
    # 软件stop 
    a = dll.GA_Stop(lMask, lOption)


    axis = 1
    lMask = (0x0001 << (axis - 1))  # 计算掩码值
    a = dll.GA_Stop(lMask, lOption)
move(10000,10000)
time.sleep(2)
stop()
move(-10000,-10000)





# # xValue = 0
# # yValue = 0
# # a=dll.GA_LnXY(1,xValue,yValue,c_double(20.5),c_double(0.9),0,0,2)

# # print('插入2维插补数据,X=50000脉冲，Y=50000脉冲,返回值:',a)
# # a = dll.GA_GetPrfPos(1, byref(dPrfPosx),1,0)
# # dValue = dPrfPosx
# # print('获取轴1脉冲位置，返回值：',a,'获取值：',dValue)
# # a = dll.GA_GetPrfPos(2, byref(dPrfPosy),1,0)
# # dValue = dPrfPosy
# # print('获取轴2脉冲位置，返回值：',a,'获取值：',dValue)

# # a=dll.GA_LnXY(1,0,0,c_double(20.5),c_double(0.9),0,0,2)
# # print('插入2维插补数据,X=0脉冲，Y=0脉冲,返回值:',a)
# a=dll.GA_CrdData(1,0,0)
# print('将数据压入控制卡')
# print('延时8秒钟')
# time.sleep(8)


a=dll.GA_Close()
print('测试结束')
