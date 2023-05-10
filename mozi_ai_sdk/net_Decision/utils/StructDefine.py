#!/usr/bin/python 3
from ctypes import *
import json


class CguVec3(Structure):
    _fields_ = [('x', c_double), ('y', c_double), ('z', c_double)]


class stObjStatus(Structure):
    _fields_ = [('iPos', CguVec3), ('iHpr', CguVec3), ('dSpd', c_double), ('dOil', c_double)]
