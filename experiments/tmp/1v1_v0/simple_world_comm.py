import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math
defense_angle=90
attack_angle=90
hit_range=0.1

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 4
        #world.damping = 1
        num_good_agents = 1
        num_adversaries = 1
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        num_food = 2
        num_forests = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.leader = True if i == 0 else False
            # agent.silent = True if i > 0 else False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.05 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 0.02 if agent.adversary else 0.02
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.0
            landmark.boundary = False
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.0
            landmark.boundary = False
        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmark in enumerate(world.forests):
            landmark.name = 'forest %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False
        world.landmarks += world.food
        world.landmarks += world.forests
        #world.landmarks += self.set_boundaries(world)  # world boundaries now penalized with negative reward
        # make initial conditions
        self.reset_world(world)
        return world

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.food):
            landmark.color = np.array([0.15, 0.15, 0.65])
        for i, landmark in enumerate(world.forests):
            landmark.color = np.array([0.6, 0.9, 0.6])
        # set random initial states
        for agent in world.agents:
            # if agent.adversary:
            #     agent.state.p_pos = np.array([-0.8,-0.8])
            #     agent.state.p_vel = np.zeros(world.dim_p)
            #     agent.state.c = np.zeros(world.dim_c)
            # else:
            #     agent.state.p_pos = np.array([+0.9, -0.9])
            #     agent.state.p_vel = np.zeros(world.dim_p)
            #     agent.state.c = np.zeros(world.dim_c)
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.food):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.forests):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1 or agent.state.p_pos[0] < -1 or agent.state.p_pos[1] > 1 or agent.state.p_pos[1] < -1:
            return True
        else:
            return False

    def will_hit(self, agent1, agent2):


        '''根据双方的攻击范围和防御范围进行判断'''
        delta_pos = agent2.state.p_pos - agent1.state.p_pos #距离向量，由攻击者指向逃脱者
        distance = np.sqrt(np.sum(np.square(delta_pos)))
        #如果靠太近，直接就坠毁，返回False
        if distance <= 1e-5:
            return False

        #agent1速度方向角，分x，y
        agent1_chi = [agent1.state.p_vel[0], agent1.state.p_vel[1]]
        #如果速度为太小，就给个固定值
        if abs(agent1.state.p_vel[0]) < 1e-5 and abs(agent1.state.p_vel[1]) < 1e-5:
            agent1_chi[0] = 0.1
            agent1_chi[1] = 0

        #agent2速度方向角，分x，y
        agent2_chi = [agent2.state.p_vel[0], agent2.state.p_vel[1]]
        if abs(agent2.state.p_vel[0]) < 1e-5 and abs(agent2.state.p_vel[1]) < 1e-5:
            agent2_chi[0] = 0.1
            agent2_chi[1] = 0

        '''算攻击者角'''
        agent1_chi_value = np.sqrt(np.sum(np.square(agent1_chi)))
        agent1_cross = (delta_pos[0] * agent1_chi[0] + delta_pos[1] * agent1_chi[1]) / (distance * agent1_chi_value)
        if agent1_cross < -1:
            agent1_cross = -1
        if agent1_cross > 1:
            agent1_cross = 1
        agent1_angle = math.acos(agent1_cross)

        '''算逃脱者的防御角'''
        agent2_chi_value = np.sqrt(np.sum(np.square(agent2_chi)))
        agent2_cross = (delta_pos[0] * agent2_chi[0] + delta_pos[1] * agent2_chi[1]) / (distance * agent2_chi_value)
        if agent2_cross < -1:
            agent2_cross = -1
        if agent2_cross > 1:
            agent2_cross = 1
        agent2_angle = math.acos(agent2_cross)

        '''如果距离小于开火距离，逃脱者在攻击区内'''
        if distance < hit_range and agent2_angle * 180 / math.pi < defense_angle / 2 and agent1_angle * 180 / math.pi < attack_angle / 2:
            return True
        return False


    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 5
        if agent.adversary:
            opponent = self.good_agents(world)
        else:
            opponent = self.adversaries(world)

        for op in opponent:
            if self.will_hit(op,agent):
                rew -= 5


        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= 2 * bound(x)

        for food in world.food:
            if self.is_collision(agent, food):
                rew += 2
        rew += 0.05 * min([np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos))) for food in world.food])

        '''wupeng forest'''
        for f in world.forests:
            if self.is_collision(agent,f):
                rew += 1
        return rew



    def adversary_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 5

        for f in world.forests:
            if self.is_collision(agent,f):
                rew -= 5


        if agent.adversary:
            opponent = self.good_agents(world)
        else:
            opponent = self.adversaries(world)

        for op in opponent:
            if self.will_hit(agent,op):
                rew += 5
            if self.will_hit(op,agent):
                rew -= 5

        return rew

    def observation2(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def observation(self, agent, world):

        defense_pos=world.forests[0].state.p_pos

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        in_forest = [np.array([-1]), np.array([-1])]
        inf1 = False
        inf2 = False
        if self.is_collision(agent, world.forests[0]):
            in_forest[0] = np.array([1])
            inf1= True
        '''wupeng'''
        # if self.is_collision(agent, world.forests[1]):
        #     in_forest[1] = np.array([1])
        #     inf2 = True

        food_pos = []
        for entity in world.food:
            if not entity.boundary:
                food_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            oth_f1 = self.is_collision(other, world.forests[0])
            # oth_f2 = self.is_collision(other, world.forests[1])
            oth_f2 = False
            if (inf1 and oth_f1) or (inf2 and oth_f2) or (not inf1 and not oth_f1 and not inf2 and not oth_f2) or agent.leader:  #without forest vis
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)
            else:
                other_pos.append([0, 0])
                if not other.adversary:
                    other_vel.append([0, 0])

        # to tell the pred when the prey are in the forest
        prey_forest = []
        ga = self.good_agents(world)
        for a in ga:
            if any([self.is_collision(a, f) for f in world.forests]):
                prey_forest.append(np.array([1]))
            else:
                prey_forest.append(np.array([-1]))
        # to tell leader when pred are in forest
        prey_forest_lead = []
        for f in world.forests:
            if any([self.is_collision(a, f) for a in ga]):
                prey_forest_lead.append(np.array([1]))
            else:
                prey_forest_lead.append(np.array([-1]))

        comm = [world.agents[0].state.c]

        if agent.adversary and not agent.leader:

            # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest + comm)
            # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest )
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos]  + other_pos + other_vel + in_forest + [defense_pos])
        if agent.leader:

            # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest + comm)
            # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest )
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos]  + other_pos + other_vel + in_forest + [defense_pos])
        else:
            # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + in_forest  )
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos]  + other_pos + other_vel + in_forest +[defense_pos] )


