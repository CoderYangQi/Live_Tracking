import cv2
import numpy as np

from ctypes import *
import ctypes
import numpy as np
import struct
import time
import math
import datetime


# from gmssl import func 
# typedef struct JogPrm
# {
#     double dAcc;
#     double dDec;
#     double dSmooth;
# }TJogPrm;
# https://www.cnblogs.com/pyse/p/8590829.html
# http://blog.sina.com.cn/s/blog_4c0cb1c00102xnw7.html
class JogPrm(Structure):
    _fields_ = [("dAcc", c_double),
                ("dDec", c_double),

                ("dSmooth", c_double)
                ]


class TTrapPrm(Structure):
    _fields_ = [("acc", c_double),
                ("dec", c_double),

                ("velStart", c_int),

                ("smoothTime", c_int)
                ]

class Move():
    def __init__(self):
        self.dll = CDLL(r'E:\yq\code\ControlModel\GAS.dll')
        print(self.dll)
        self.dll.GA_StartDebugLog(1)
        print('测试开始')
        a=self.dll.GA_OpenByIP(b'192.168.0.2',b'192.168.0.1',0,0)
        #a=dll.GA_Open(1,"COM1")
        print('打开板卡GA_Open返回值:',a)

        a=self.dll. GA_Reset()
        print('复位板卡GA_Reset返回值:',a)

        a=self.dll.GA_EncOff(1)
        print('关闭轴1编码器:',a)

        a=self.dll.GA_ZeroPos(1,3)
        print('清零轴1零位，返回值:',a)

    
    def moveXY(self, Xvalue, Yvalue):
        a=self.dll.GA_EncOff(1)
        a=self.dll.GA_EncOff(2)
        print(f'关闭轴{1}编码器:',a)

        a=self.dll.GA_ZeroPos(1,3)
        print(f'清零轴{1}零位，返回值:',a)

        #  轴1 的状态
        a=self.dll.GA_AxisOn(1)
        print('使能轴1返回值:',a)

        a=self.dll.GA_PrfTrap(1)
        print('设置轴1进入点位模式，返回值:',a)

        a=self.dll.GA_SetTrapPrmSingle(1,c_double(1.0),c_double(1.0),c_double(0.0),0)
        print('设置轴1点位运动参数，返回值:',a)

        # while 1:
        dPrfPos = c_double(0.0)
        print('打开输出口Y5')
        # a=self.dll.GA_SetExtDoBit(0, 5, 1)
        a=self.dll.GA_SetPos(1,Xvalue)
        print('设置轴1运动目标位置为20000脉冲的位置，返回值:',a)
        a=self.dll.GA_SetVel(1,c_double(7.5))
        print('设置轴1运动速度为7.5脉冲/毫秒，返回值:',a)
        a=self.dll.GA_Update(1)
        print('启动轴1运动')
        print('延时5秒钟')
        time.sleep(5)

        a = self.dll.GA_GetPrfPos(1, byref(dPrfPos),1,0)
        dValue = dPrfPos
        print('获取轴1脉冲位置，返回值：',a,'获取值：',dValue)

        lSts = c_long(0)
        a = self.dll.GA_GetSts(1, byref(lSts),1,0)
        print('获取轴1状态，返回值：',a,'获取值：',lSts)

        #  轴2 的状态
        a=self.dll.GA_AxisOn(2)
        print('使能轴2返回值:',a)

        a=self.dll.GA_PrfTrap(2)
        print('设置轴2进入点位模式，返回值:',a)

        a=self.dll.GA_SetTrapPrmSingle(2,c_double(1.0),c_double(1.0),c_double(0.0),0)
        print('设置轴2点位运动参数，返回值:',a)

        # while 1:
        dPrfPos = c_double(0.0)
        print('打开输出口Y5')
        # a=self.dll.GA_SetExtDoBit(0, 5, 1)
        a=self.dll.GA_SetPos(2,Xvalue)
        print(f'设置轴2运动目标位置为 {Yvalue}脉冲的位置，返回值:',a)
        a=self.dll.GA_SetVel(2,c_double(7.5))
        print('设置轴2运动速度为7.5脉冲/毫秒，返回值:',a)
        a=self.dll.GA_Update(2)
        print('启动轴2运动')
        print('延时5秒钟')
        time.sleep(5)

        a = self.dll.GA_GetPrfPos(2, byref(dPrfPos),1,0)
        dValue = dPrfPos
        print('获取轴2脉冲位置，返回值：',a,'获取值：',dValue)

        lSts = c_long(0)
        a = self.dll.GA_GetSts(2, byref(lSts),1,0)
        print('获取轴2状态，返回值：',a,'获取值：',lSts)



            
        # a=self.dll.GA_Close()
        # print('测试结束')


if __name__ == "__main__":
    mymove = Move()

