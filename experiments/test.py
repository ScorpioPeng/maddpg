import argparse
import math
import os

import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

from RedisInterface import PubSub
import redis
import json
from JWD import JWD2XY, XY2JWD, fetch_vel


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_world_comm", help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int, default=250, help="maximum episode length")  # 每一轮游戏里的步数
    parser.add_argument("--num_episodes", type=int, default=120000, help="number of episodes")  # 总共游戏次数
    parser.add_argument("--num_adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good_policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv_policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")  # 学习率
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")  # 奖励折扣率
    parser.add_argument("--batch_size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num_units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp_name", type=str, default='exp1', help="name of the experiment")
    parser.add_argument("--save_dir", type=str, default="./tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save_rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load_dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark_iters", type=int, default=5000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark_dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots_dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv  #
    import multiagent.scenarios as scenarios  # 场景文件夹，里面有多种不同的场景

    # load scenario from script 创建该场景的类
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world 调用场景类里面的make world初始化
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []  # 每个agent有一个trainer
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session() as sess:
        # print(tf.get_default_session())
        # Create environment创建环境
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers ,env.n为agent的数量
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        '''obs_shape_n:[(35,), (35,), (35,), (35,), (35,), (35,)]'''
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        '''wupeng json接收初始位置
        {'entityInfo': [{"agent":10,"id":1001,"lat":21,"lon":126},{"agent":20,"id":1001,"lat":21,"lon":126}]}
        '''

        obj = PubSub('localhost', 6379, 1)
        redis_sub = obj.subscribe('REDIS_TOPIC_ENTITYINFO_TO_RLMODEL')
        msg = redis_sub.parse_response()
        msg = msg[2].decode()
        my_json = json.loads(msg)
        obs_n = handle_json(my_json)  # 这步是接受平台初始的obs

        # obs_n = env.reset()  #原来每个agent的局部观察
        print('Starting testing...')
        while True:  # 每一次循环就是移动一步
            # get action
            '''action_n是要经过网络训练得出的'''
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            '''根据action_n和上一步'''
            actionJson = publish_action(my_json, action_n)
            obj.publish('REDIS_TOPIC_MAKING_DECISION', json.dumps(actionJson))
            '''action发过去之后立刻开始等待下一个obs'''
            # environment step
            # new_obs_n, rew_n, done_n, info_n = env.step(action_n) #采用动作a后，生成s_,r
            msg = redis_sub.parse_response()
            msg = msg[2].decode()
            my_json = json.loads(msg)
            new_obs_n = handle_json(my_json)
            obs_n = new_obs_n

            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue


def handle_json(my_json):
    entitylist = my_json['entityInfo']  # [{"agent":10,"id":1001,"lat":21,"lon":126},{"agent":20,"id":1001,"lat":21,"lon":126}]
    defense = my_json['defense']
    dx, dy = JWD2XY(defense["lon"], defense["lat"], 113.068, 17.0594)  # 位置
    defense_pos = np.array([dx, dy])
    for e in range(2):
        lon = entitylist[e]["lon"]
        lat = entitylist[e]["lat"]
        speed = entitylist[e]["speed"]
        angle = entitylist[e]["angle"] * np.pi / 180
        x, y = JWD2XY(lon, lat, 113.068, 17.0594)  # 位置
        xspeed, yspeed = fetch_vel(speed, angle)  # 速度
        in_forest = entitylist[e]["in_forest"]  # 是否在防御区里面
        if in_forest == 0:
            inf = np.array([-1.0, -1.0])
        else:
            inf = np.array([1.0, -1.0])

        newjson = []
        agent = {}
        agent["id"] = entitylist[e]["id"]
        agent["pos"] = np.array([x, y])
        agent["vel"] = np.array([xspeed, yspeed])
        agent["in_forest"] = np.array(inf)
        agent["self_chi"] = np.array(math.atan2(yspeed,xspeed))
        newjson.append(agent)  # 这一步把传过来的json转换成符合gym的json数据

    obs=[]
    n = len(newjson)
    chi=[]
    for i in range(n):
        chi.append(newjson[i]["self_chi"])

    for i in range(n):
        other_pos = []
        other_vel = []
        for j in range(n):
            if i==j:continue
            other_pos.append(newjson[j]["pos"] - newjson[i]["pos"])
            other_vel.append(newjson[j]["vel"] - newjson[i]["vel"])
        obs_i = np.concatenate([newjson[i]["vel"]] + [newjson[i]["pos"]] + other_pos + other_vel + [newjson[i]["in_forest"]] + [defense_pos] + chi)
        obs.append(obs_i)
    return obs

def publish_action(my_json, action):
    entitylist = my_json['entityInfo']
    InfoJson = {}  # {'actionInfo':[{},{},{}]}
    Info = []
    for i in range(len(action)):
        jsonInfo = {}
        jsonInfo["id"] = entitylist[i]["id"]
        jsonInfo["xforce"] = (action[i][1] - action[i][2]) * 10000
        jsonInfo["yforce"] = (action[i][3] - action[i][4]) * 10000
        Info.append(jsonInfo)
    InfoJson["actionInfo"] = Info
    return InfoJson


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
