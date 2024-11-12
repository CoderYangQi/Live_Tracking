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
#a=dll.GA_Open(0,"192.168.0.200")
#a=dll.GA_Open(1,"COM1")
print('打开板卡GA_Open返回值:',a)

a=dll. GA_Reset()
print('复位板卡GA_Reset返回值:',a)
dPrfPos = c_double(0.0)

axis = 3
value = 200000
dPrfPosz = c_double(0)
a=dll.GA_EncOff(axis)
print(f'关闭轴{axis}编码器:',a)

a=dll.GA_ZeroPos(1,8)
print('清零轴1零位，返回值:',a)

a=dll.GA_AxisOn(axis)
print('使能轴1返回值:',a)

a=dll.GA_PrfTrap(axis)
print('设置轴1进入点位模式，返回值:',a)

a=dll.GA_SetTrapPrmSingle(axis,c_double(1.0),c_double(1.0),c_double(0.0),0)
print('设置轴1点位运动参数，返回值:',a)

print('打开输出口Y5')
# a=dll.GA_SetExtDoBit(0, 5, 1)
a=dll.GA_SetPos(axis,value)
print('设置轴1运动目标位置为20000脉冲的位置，返回值:',a)
a=dll.GA_SetVel(axis,c_double(7.5))
print('设置轴1运动速度为7.5脉冲/毫秒，返回值:',a)
a=dll.GA_Update(2**(axis - 1))
time.sleep(2)
lMask = (0x0001 << (axis - 1))  # 计算掩码值
lOption = 0  # 根据函数要求设置此参数
a = dll.GA_Stop(lMask, lOption)

# get pos
a = dll.GA_GetPrfPos(axis, byref(dPrfPosz),1,0)
dValuez = dPrfPosz
print('获取轴1脉冲位置，返回值：',a,'获取值：',dValuez)
time.sleep(2)
a=dll.GA_SetPos(axis,value)
print('设置轴1运动目标位置为20000脉冲的位置，返回值:',a)
a=dll.GA_SetVel(axis,c_double(7.5))
print('设置轴1运动速度为7.5脉冲/毫秒，返回值:',a)
a=dll.GA_Update(2**(axis - 1))
print('启动轴1运动')
print('延时5秒钟')

time.sleep(2)
a = dll.GA_Stop(lMask, lOption)
# get pos
a = dll.GA_GetPrfPos(axis, byref(dPrfPosz),1,0)
dValuez = dPrfPosz
print('获取轴1脉冲位置，返回值：',a,'获取值：',dValuez)

print("finished")

# time.sleep(5)
# a = dll.GA_GetPrfPos(axis, byref(dPrfPos),1,0)
# dValue = dPrfPos
# print('获取轴1脉冲位置，返回值：',a,'获取值：',dValue)

# lSts = c_long(0)
# a = dll.GA_GetSts(axis, byref(lSts),1,0)
# print('获取轴1状态，返回值：',a,'获取值：',lSts)



# print('关闭输出口Y5')
# a = dll.GA_SetExtDoBit(0, 5, 0)
# a=dll.GA_SetPos(axis,0)
# print('设置轴1运动目标位置为0脉冲的位置，返回值:',a)
# a=dll.GA_Update(axis)
# print('启动轴1运动')
# print('延时5秒钟')
# # time.sleep(5)
# dPrfPos = c_double(0)
# a = dll.GA_GetPrfPos(1, byref(dPrfPos),1,0)
# dValue = dPrfPos
# print('获取轴1脉冲位置，返回值：',a,'获取值：',dValue)
    
a=dll.GA_Close()
print('测试结束')
