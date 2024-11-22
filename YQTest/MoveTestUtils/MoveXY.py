from ctypes import *
import ctypes
#import numpy as np
import struct 
import time
import math

def MoveXY():
    
    dll = CDLL(r"F:\yq\code\Live_Tracking\YQTest\GAS.dll")
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
    a=dll.GA_AxisOn(1)
    a=dll.GA_AxisOn(2)
    a=dll.GA_AxisOn(3)
    def move(axis, Value, speed):
            
        lMask = (0x0001 << (axis - 1))  # 计算掩码值
        lOption = 0  # 根据函数要求设置此参数
        # 软件stop 
        a = dll.GA_Stop(lMask, lOption)

        a=dll.GA_PrfTrap(axis)
        print(f'设置轴{axis}进入点位模式，返回值:',a)

        a=dll.GA_SetTrapPrmSingle(axis,c_double(1.0),c_double(1.0),c_double(0.0),0)
        print(f'设置轴{axis}点位运动参数，返回值:',a)
        a=dll.GA_SetPos(axis,Value)
        print(f'设置轴{axis}运动目标位置为{Value}脉冲的位置，返回值:',a)
        a=dll.GA_SetVel(axis,c_double(speed))
        print(f'设置轴{axis}运动速度为{speed}脉冲/毫秒，返回值:',a)
        a=dll.GA_Update(2**(axis - 1))
    # test axis x
    axis = 1;speed = 5; Value = 80000;
    move(axis, Value, speed)

    # test axis y

    axis = 2;speed = 5; Value = -100000;
    move(axis, Value, speed)
    
    a=dll.GA_Close()
    print('测试结束')
if __name__ == "__main__":
    MoveXY()