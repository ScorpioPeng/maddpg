from JWD import JWD2XY,XY2JWD,fetch_vel
import numpy as np

def handle_json(my_json):

    newjson=[]
    obs=[]
    entitylist = my_json['entityInfo']#[{"agent":10,"id":1001,"lat":21,"lon":126},{"agent":20,"id":1001,"lat":21,"lon":126}]
    n = len(entitylist)
    for e in range(n):
        agent={}
        lon = entitylist[e]["lon"]
        lat = entitylist[e]["lat"]
        speed = entitylist[e]["speed"]
        angle = entitylist[e]["angle"]*np.pi/360
        x, y = JWD2XY(lon, lat,0,0)#位置
        xspeed, yspeed = fetch_vel(speed, angle)#速度
        agent["id"]=entitylist[e]["id"]
        agent["pos"]=np.array([x,y])
        agent["vel"]=np.array([xspeed,yspeed])
        agent["chi"]=np.array(angle)
        newjson.append(agent)
    for i in range(n):
        other_pos = []
        other_vel = []
        other_chi = []
        for k in range(n):#10
            if newjson[k]["id"]!=newjson[i]["id"]:
                other_pos.append(newjson[k]["pos"]-newjson[i]["pos"])
        for k in range(n):#10
            if newjson[k]["id"]!=newjson[i]["id"]:
                other_vel.append(newjson[k]["vel"])
        for k in range(n):#5
            if newjson[k]["id"]!=newjson[i]["id"]:
                other_chi.append(newjson[k]["chi"])
        action_number = [np.zeros(5)]
        obs_tmp = np.concatenate([newjson[i]["vel"]]+[newjson[i]["pos"]]+ other_pos + other_vel + [[newjson[i]["chi"]]] + [other_chi]+action_number)
        obs.append(obs_tmp)
    return obs

#这是写给maddpg的
def handle_json2(my_json):
    newjson=[]
    obs=[]
    entitylist = my_json['entityInfo'] #[{"agent":10,"id":1001,"lat":21,"lon":126},{"agent":20,"id":1001,"lat":21,"lon":126}]
    defense=my_json['defense']
    dx,dy=JWD2XY(defense["lon"], defense["lat"],113.068,17.0594)#位置
    defense_pos=np.array([dx,dy])
    for e in range(2):
        agent={}
        lon = entitylist[e]["lon"]
        lat = entitylist[e]["lat"]
        speed = entitylist[e]["speed"]
        angle = entitylist[e]["angle"]*np.pi/180
        x, y = JWD2XY(lon, lat,113.068,17.0594)#位置
        xspeed, yspeed = fetch_vel(speed, angle)#速度
        in_forest = entitylist[e]["in_forest"]  # 是否在防御区里面
        if in_forest == 0:
            inf=np.array([-1.0,-1.0])
        else:
            inf=np.array([1.0,-1.0])
        agent["id"]=entitylist[e]["id"]
        agent["pos"]=np.array([x,y])
        agent["vel"]=np.array([xspeed,yspeed])
        agent["in_forest"]=np.array(inf)
        newjson.append(agent)  #这一步把传过来的json转换成符合gym的json数据

    #leader
    other_pos0 = []
    other_vel0 = []

    #good agent
    other_pos1 = []
    other_vel1 = []

    other_pos0.append(newjson[1]["pos"]-newjson[0]["pos"])
    other_pos1.append(newjson[0]["pos"]-newjson[1]["pos"])
    other_vel0.append(newjson[1]["vel"])
    obs0 = np.concatenate([newjson[0]["vel"]]+[newjson[0]["pos"]]+ other_pos0 + other_vel0 + [newjson[0]["in_forest"]]+ [defense_pos] )
    obs1 = np.concatenate([newjson[1]["vel"]]+[newjson[1]["pos"]]+ other_pos1 + other_vel1 + [newjson[1]["in_forest"]]+ [defense_pos])
    obs.append(obs0)
    obs.append(obs1)
    return obs
'''
[[0.01974902, 0.7713059 , 0.1075811 , 0.04185394, 0.05951015],
[0.6618421 , 0.0514295 , 0.12686594, 0.13395205, 0.02591034],
[0.0866556 , 0.02502446, 0.2823804 , 0.592616  , 0.01332349],
[0.2096966 , 0.15619908, 0.11971663, 0.47036523, 0.04402252],
[0.04314107, 0.7988486 , 0.13247246, 0.01095159, 0.01458634],
[0.02745177, 0.09874075, 0.69300413, 0.02764952, 0.15315379]]
'''
def publish_action(my_json,action):
    entitylist = my_json['entityInfo']
    InfoJson={}#{'actionInfo':[{},{},{}]}
    Info=[]
    for i in range(len(action)):
        jsonInfo={}
        jsonInfo["id"]=entitylist[i]["id"]
        jsonInfo["xforce"]=action[i][1]-action[i][2]
        jsonInfo["yforce"]=action[i][3]-action[i][4]
        Info.append(jsonInfo)
    InfoJson["actionInfo"]=Info
    return InfoJson



if __name__ == '__main__':
    a={'entityInfo': [{"agent":-1,"id":1001,"lat":21,"lon":126,"speed":10,"angle":30,"in_forest":0},
                      {"agent":0,"id":1002,"lat":21,"lon":126,"speed":10,"angle":30,"in_forest":1},
                      {"agent":1,"id":1003,"lat":21,"lon":126,"speed":10,"angle":30,"in_forest":0},
                      {"agent":2,"id":1004,"lat":21,"lon":126,"speed":10,"angle":30,"in_forest":0},
                      {"agent":10,"id":1005,"lat":21,"lon":126,"speed":10,"angle":30,"in_forest":1},
                      {"agent":20,"id":1006,"lat":21,"lon":126,"speed":10,"angle":30,"in_forest":1}],
       'defense':{"lon":123,"lat":32}}
    o=handle_json2(a)
    print(o)
    action=[[0.01974902, 0.7713059 , 0.1075811 , 0.04185394, 0.05951015],
            [0.6618421 , 0.0514295 , 0.12686594, 0.13395205, 0.02591034],
            [0.0866556 , 0.02502446, 0.2823804 , 0.592616  , 0.01332349],
            [0.2096966 , 0.15619908, 0.11971663, 0.47036523, 0.04402252],
            [0.04314107, 0.7988486 , 0.13247246, 0.01095159, 0.01458634],
            [0.02745177, 0.09874075, 0.69300413, 0.02764952, 0.15315379]]
    print(publish_action(a,action))
