import math
gh= math.pi/180

def C2PI(c):
    tmp=c%(2*math.pi)
    if tmp >=0 and tmp<=2*math.pi:
        return tmp
    else:
        tmp = tmp +2*math.pi
    return tmp

#度转弧度
def Degree2Radian(t):
    return (t)*math.pi/180


#弧度转度
def Radian2Degree(t):
    return (t)*180/math.pi



'''
@brief 已知两点的经纬度，计算距离和方位
@param A1
@param A2
@param W1
@param W2
@param R 两点之间的距离，单位：米
@param B 两点之间的角度
'''
def ComputeDis(A1,A2,W1,W2):
    aa = 6378137
    alfa = 1/298.257223563
    A1 = Degree2Radian(A1)
    A2 = Degree2Radian(A2)
    W1 = Degree2Radian(W1)
    W2 = Degree2Radian(W2)
    D = math.acos(math.sin(W1)*math.sin(W2)+math.cos(W1)*math.cos(W2)*math.cos(A2-A1))
    temp = (3*math.sin(D)-D)*math.pow(math.sin(W1)+math.sin(W2),2)/(1+math.cos(D))-(3*math.sin(D)+D)*math.pow(math.sin(W1)-math.sin(W2),2)/(1-math.cos(D))
    if A1 == A2 and W1==W2:
        R=0
    else:
        R=math.fabs((aa*D+aa*alfa*temp/4))
    temp1=math.cos(W1)*math.tan(W2)-math.sin(W1)*math.cos(A2-A1)
    if temp1 !=0:
        B = C2PI(math.atan2(math.sin(A2-A1),temp1))
    else:
        B=math.pi/2
    B=Radian2Degree(B)
    return R,B

def XY2JWD(xT,yT,sublong0,sublat0):
    xw0=0
    yw0=0
    Tdlong=0.0
    Tdlat=0.0
    sublat0=sublat0*gh
    sublong0=sublong0*gh
    R=6378137
    a=1.0/298.257223563
    e2=a*(2.0-a)
    Rm=R*(1-e2)/math.pow((1-e2*math.sin(sublat0)*math.sin(sublat0)),1.5)
    Rn=R/math.pow((1-e2*math.sin(sublat0)*math.sin(sublat0)),0.5)
    x1=xT-xw0
    y1=yT-yw0

    D0=math.sqrt(x1*x1+y1*y1)
    beta=C2PI(math.atan2(x1,y1))
    Tdlat=(sublat0+D0*math.cos(beta)/Rm)*180/math.pi
    Tdlong=(sublong0+D0*math.sin(beta)/(Rn*math.cos(Tdlat*(gh))))*180/math.pi
    while (Tdlat > 90):
        Tdlat = 180 - Tdlat
    while (Tdlat < -90):
        Tdlat = -180 - Tdlat

    while (Tdlong > 180):
        Tdlong = Tdlong - 180 * 2.0
    while (Tdlong < -180):
        Tdlong = Tdlong + 180 * 2.0
    return Tdlong,Tdlat


'''
函数功能:将经纬度信息转换为相对坐标
参数

Tdlong-经度 单位度
Tdlat-纬度 单位度
sublong0-原点对应的经度 单位度
sublat0-原点对应的本艇纬度 单位度

输出
xT-横坐标
yT-纵坐标
'''
def JWD2XY(Tdlong,Tdlat,sublong0,sublat0):
    xT,B1 = ComputeDis(sublong0,Tdlong,sublat0,sublat0)
    if xT!=0:
        if B1>180 and B1 <360:
            xT = xT * (-1)
    yT,B2 = ComputeDis(sublong0,sublong0,sublat0,Tdlat)
    if yT!=0:
        if(B2>90 and B2<270):
            yT= yT*(-1)
    return xT/10000,yT/10000
'''返回数据还需根据缩放比例进行处理'''

def fetch_vel(speed,angle):
    xspeed=speed*math.sin(angle)
    yspeed=speed*math.cos(angle)
    return xspeed/10000,yspeed/10000