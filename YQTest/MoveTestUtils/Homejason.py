from ctypes import *
import ctypes
#import numpy as np
import struct 
import time
import math
from ctypes import c_short, c_long, c_ulong, POINTER

# MC_GetSts(short nAxisNum,long *pSts,short nCount=1,unsigned long *pClock=NULL
# 定义状态位标志
AXIS_STATUS_ESTOP = 0x00000001          # 急停
AXIS_STATUS_POS_SOFT_LIMIT = 0x00000004 # 正软限位触发标志
AXIS_STATUS_NEG_SOFT_LIMIT = 0x00000008 # 负软位触发标志
AXIS_STATUS_FOLLOW_ERR = 0x00000010     # 误差过大
AXIS_STATUS_POS_HARD_LIMIT = 0x00000020 # 正硬限位触发标志
AXIS_STATUS_NEG_HARD_LIMIT = 0x00000040 # 负硬限位触发标志
AXIS_STATUS_IO_SMS_STOP = 0x00000080    # 保留
AXIS_STATUS_IO_EMG_STOP = 0x00000100    # 保留
AXIS_STATUS_RUNNING = 0x00000400        # 规划运动标志
AXIS_STATUS_ARRIVE = 0x00000800         # 电机到位
AXIS_STATUS_HOME_RUNNING = 0x00001000   # 正在回零
AXIS_STATUS_HOME_SUCESS = 0x00002000    # 回零成功
AXIS_STATUS_HOME_SWITCH = 0x00004000    # 零位信号
AXIS_STATUS_GEAR_START = 0x00010000     # 电子齿轮开始啮合
AXIS_STATUS_GEAR_FINISH = 0x00020000    # 电子齿轮完成啮合
# 状态位对应的描述
STATUS_DESCRIPTIONS = {
    AXIS_STATUS_ESTOP: "急停",
    AXIS_STATUS_POS_SOFT_LIMIT: "正软限位触发标志",
    AXIS_STATUS_NEG_SOFT_LIMIT: "负软位触发标志",
    AXIS_STATUS_FOLLOW_ERR: "误差过大",
    AXIS_STATUS_POS_HARD_LIMIT: "正硬限位触发标志",
    AXIS_STATUS_NEG_HARD_LIMIT: "负硬限位触发标志",
    AXIS_STATUS_IO_SMS_STOP: "保留：SMS 停止",
    AXIS_STATUS_IO_EMG_STOP: "保留：紧急停止",
    AXIS_STATUS_RUNNING: "规划运动标志",
    AXIS_STATUS_ARRIVE: "电机到位",
    AXIS_STATUS_HOME_RUNNING: "正在回零",
    AXIS_STATUS_HOME_SUCESS: "回零成功",
    AXIS_STATUS_HOME_SWITCH: "零位信号",
    AXIS_STATUS_GEAR_START: "电子齿轮开始啮合",
    AXIS_STATUS_GEAR_FINISH: "电子齿轮完成啮合",
}
def parse_axis_status(status):
    """
    解析 32 位轴状态字为具体描述。
    
    :param status: 32 位状态字 (int)
    :return: 各状态对应的描述列表
    """
    results = []
    successFlag = False
    for bit, description in STATUS_DESCRIPTIONS.items():
        if status & AXIS_STATUS_HOME_SUCESS:
            successFlag = True
        if status & bit:  # 检查当前位是否被设置
            results.append(description)
    return results, successFlag

def GoHome():
    
    dll = CDLL('F:\yq\code\PythonHomeProject\GAS.dll')
    print(dll)

    dll.GA_StartDebugLog(1)

    print('测试开始')
    a=dll.GA_OpenByIP(b'192.168.0.2',b'192.168.0.1',0,0)
    #a=dll.GA_Open(0,"192.168.0.200")
    print('打开板卡GA_Open返回值:',a)

    a=dll. GA_Reset()
    print('复位板卡GA_Reset返回值:',a)
    a=dll.GA_AxisOn(1)
    a=dll.GA_AxisOn(2)
    # a=self.dll.GA_AxisOn(2)
    # a=self.dll.GA_AxisOn(3)



    #启动轴1回零（注意如果某一次回零失败，需要GA_HomeStop停止回零，否则轴不能移动）
    # 准备参数
    nAxisNum = c_short(1)  # 示例轴号
    nCount = c_short(2)    # 查询的轴状态数量

    # 分配内存给输出参数
    pSts = (c_long * nCount.value)()  # 创建长整型数组存储状态
    pClock = ctypes.POINTER(c_ulong)()





    #设置回零参数
    #第1个参数轴号，1代表第一个轴
    #第2个参数回零模式，1代表Home原点回零
    #第3个参数回零方向，0代表负向回零，1代表正向回零
    #第4个参数回零偏移，代表回零完成后，再走一个固定偏移量作为零点，单位脉冲，通常为0
    #第5个参数回零快移速度，单位脉冲/毫秒，可以为小数，取值范围0.01~200，通常为5~50
    #第6个参数回零定位速度，单位脉冲/毫秒，可以为小数，取值范围0.01~200，通常为1~5
    #第7个参数回零Index速度，单位脉冲/毫秒，通常为1
    #第8个参数回零加速度，单位脉冲/毫秒/毫秒，取值范围0.01~5，通常为1，可以为小数
    # a=dll. GA_HomeSetPrmSingle(axis,1,0,0,c_double(1.0),c_double(1.0),c_double(1.0),c_double(0.1),0,0,0)


    speed = 3
    axisX = 1
    a=dll.GA_HomeSetPrmSingle(axisX,1,0,0,c_double(speed),c_double(1.0),c_double(1.0),c_double(0.1),0,0,0)
    a=dll. GA_HomeStart(axisX)
    axisY = 2
    a=dll.GA_HomeSetPrmSingle(axisY,1,1,0,c_double(speed),c_double(1.0),c_double(1.0),c_double(0.1),0,0,0)
    a=dll. GA_HomeStart(axisY)
    while 1:
        dll.GA_GetSts(nAxisNum,pSts,nCount,pClock)
        pSts_value = 0x00000000
        ct = 0
        nums = 2
        for i in range(nums):
            axisNum = i + 1
            pSts_value = pSts[i]
            parsed_status, successFlag = parse_axis_status(pSts_value)
            if successFlag:
                ct += 1
            print(f"{axisNum}轴状态字: {bin(pSts_value)}")
            print("解析结果:")
            for desc in parsed_status:
                print(f"{axisNum}- {desc}")
        if ct == nums:
            print("all successful")
            break
        time.sleep(0.5)
        print()
        # print('打开输出口Y5')
        # a=dll.GA_SetExtDoBit(0, 5, 1)
        # print('延时1秒钟')
        # time.sleep(1)
        # print('关闭输出口Y5')
        # a = dll.GA_SetExtDoBit(0, 5, 0)
        # print('延时1秒钟')
        # time.sleep(1)

        # dPrfPos = c_double(0)
        # a = dll.GA_GetPrfPos(1, byref(dPrfPos),1)

        # dValue = dPrfPos
        # print('获取轴1脉冲位置，返回值：',a,'获取值：%.3f',dValue)

    a=dll.GA_Close()
    print('测试结束')
if __name__ == "__main__":
    GoHome()
