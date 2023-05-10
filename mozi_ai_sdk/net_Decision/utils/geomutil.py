#!/usr/bin/python 3
from mozi_ai_sdk.net_Decision.utils import StructDefine
import math

DBL_EPSILON = 2.2204460492503131e-016
# 坐标中心，北天东坐标系
#### 2020 before 1st August Display client
# g_TranslateCenter = StructDefine.CguVec3(86.873893,42.274033, 0.0)
# g_TranslateCenter = StructDefine.CguVec3(86.873893,42.304033,0.0)
## 86.873893,42.304033,0.0 ground height 1185-1190, had better set 1190

## New release
## Longitude, Latitude, Altitude(ground over sea level)
# g_TranslateCenter = StructDefine.CguVec3(22.7, 134.17, 0)
g_TranslateCenter = StructDefine.CguVec3(19.33, 130, 0)
test_point = StructDefine.CguVec3(20.33, 131, 0)


def point_convert(point):
    lat = float(point[0])
    lon = float(point[1])
    return StructDefine.CguVec3(lat, lon, 0)

# 该函数将一个角限制在[0,2PI]，输入为弧度值
def CheckAngle(dAngle):
    dNewAngle = math.fmod(dAngle, 2 * math.pi)
    if dNewAngle < 0:
        return dNewAngle + 2 * math.pi
    return dNewAngle


# 该函数将角度转为弧度
def A2R(x):
    return x * math.pi / 180.0


# 该函数将弧度转为角度
def R2A(x):
    return x * 180.0 / math.pi


# 该函数将经纬高转为XYZ
def ConvertLLA2XYZ(iLLA):
    iLLA = point_convert(iLLA)
    earthRadius = 6371000.0
    iLLA.x = A2R(iLLA.x)
    iLLA.y = A2R(iLLA.y)
    iCenter = StructDefine.CguVec3()
    iCenter.x = A2R(g_TranslateCenter.x)
    iCenter.y = A2R(g_TranslateCenter.y)
    xyz = StructDefine.CguVec3()
    delta_lon = iLLA.x - iCenter.x
    tmp = math.sin(iLLA.y) * math.sin(iCenter.y) + math.cos(iLLA.y) * math.cos(iCenter.y) * math.cos(delta_lon)
    xyz.x = (earthRadius * math.cos(iLLA.y) * math.sin(delta_lon)) / (-tmp * 1000)
    xyz.y = (earthRadius * (math.sin(iLLA.y) * math.cos(iCenter.y) - math.cos(iLLA.y) * math.sin(iCenter.y) * math.cos(
        delta_lon))) / (tmp * 1000)
    xyz.z = iLLA.z
    return xyz


# 该函数将XYZ转为经纬高
def ConvertXYZ2LLA(iXYZ, Scale=1):
    ## the base scale is 1 for 1 meter in real world
    earthRadius = 6371000.0
    iXYZ.x = iXYZ.x * Scale / earthRadius
    iXYZ.y = iXYZ.y * Scale / earthRadius
    iCenter = StructDefine.CguVec3()
    iCenter.x = A2R(g_TranslateCenter.x)
    iCenter.y = A2R(g_TranslateCenter.y)
    lla = StructDefine.CguVec3()
    tmp = math.sqrt(iXYZ.x * iXYZ.x + iXYZ.y * iXYZ.y)
    if tmp > -DBL_EPSILON and tmp < DBL_EPSILON:
        iCenter.x = R2A(iCenter.x)
        iCenter.y = R2A(iCenter.y)
        iCenter.z = iXYZ.z
        return iCenter
    c = math.atan(tmp)
    lla.x = iCenter.x + math.atan(
        iXYZ.x * math.sin(c) / (tmp * math.cos(iCenter.y) * math.cos(c) - iXYZ.y * math.sin(iCenter.y) * math.sin(c)))
    lla.y = math.asin(math.cos(c) * math.sin(iCenter.y) + iXYZ.y * math.sin(c) * math.cos(iCenter.y) / tmp)
    lla.x = R2A(lla.x)
    lla.y = R2A(lla.y)
    lla.z = iXYZ.z
    return lla


# 该函数将iPos从经纬高转为XYZ，iHpr从角度转为弧度
def ShowToCalc(iObjStatus):
    iObjStatus.iPos = ConvertLLA2XYZ(iObjStatus.iPos)
    iObjStatus.iHpr.x = CheckAngle(A2R(90.0 - iObjStatus.iHpr.x))
    iObjStatus.iHpr.y = A2R(iObjStatus.iHpr.y)
    iObjStatus.iHpr.z = A2R(iObjStatus.iHpr.z)
    return iObjStatus


# 该函数将iPos从XYZ转为经纬高，iHpr从弧度转为角度
def CalcToShow(iObjStatus):
    iObjStatus.iPos = ConvertXYZ2LLA(iObjStatus)
    iObjStatus.iHpr.x = R2A(CheckAngle(0.5 * math.pi - iObjStatus.iHpr.x))
    iObjStatus.iHpr.y = R2A(iObjStatus.iHpr.y)
    iObjStatus.iHpr.z = R2A(iObjStatus.iHpr.z)
    return iObjStatus
