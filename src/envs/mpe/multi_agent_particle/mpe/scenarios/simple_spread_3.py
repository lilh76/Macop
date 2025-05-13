import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario

EPS = 1e-3

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.2
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.2
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.state.p_pos = np.array([np.random.uniform(1 - EPS, 1 + EPS), np.random.uniform(0 - EPS, 0 + EPS)])
            elif i == 1:
                agent.state.p_pos = np.array([np.random.uniform(-0.5 - EPS, -0.5 + EPS), np.random.uniform(0.87 - EPS, 0.87 + EPS)])
            elif i == 2:
                agent.state.p_pos = np.array([np.random.uniform(-0.5 - EPS, -0.5 + EPS), np.random.uniform(-0.87 - EPS, -0.87 + EPS)])
            agent.state.p_vel = np.zeros(world.dim_p)
            # agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.state.p_pos = np.array([np.random.uniform(0.5 - EPS, 0.5 + EPS), np.random.uniform(0.87 - EPS, 0.87 + EPS)])
            elif i == 1:
                landmark.state.p_pos = np.array([np.random.uniform(-1 - EPS, -1 + EPS), np.random.uniform(0 - EPS, 0 + EPS)])
            elif i == 2:
                landmark.state.p_pos = np.array([np.random.uniform(0.5 - EPS, 0.5 + EPS), np.random.uniform(-0.87 - EPS, -0.87 + EPS)])
            landmark.state.p_vel = np.zeros(world.dim_p)

        self.is_done = False

        # print(1, [landmark.state.p_pos for landmark in world.landmarks])

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_close(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity1.size + entity2.size
        return True if dist < dist_min * 1.7 else False

    # def reward(self, agent, world):
    #     # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #     rew = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
    #         rew -= min(dists)
    #     # if agent.collide:
    #     #     for a in world.agents:
    #     #         if self.is_collision(a, agent):
    #     #             rew -= 1
    #     return rew
    
    # def done(self, agent, world):
    #     return False

    def reward(self, agent, world):
        rew = 0
        # if abs(agent.state.p_pos).max() > 1:
        #     rew -= 0.01
        if self.done(agent, world) and not self.is_done:
            # print(2, [agent.state.p_pos for agent in world.agents])
            self.is_done = True
            rew += 1.0
        return rew

    def done(self, agent, world):
        occupied_n_landmark = []
        for agent in world.agents:
            for i, landmark in enumerate(world.landmarks):
                if self.is_close(landmark, agent):
                    occupied_n_landmark.append(i)
        if len(set(occupied_n_landmark)) == len(world.agents):
            return True
        else:
            return False
        # for landmark in world.landmarks:
        #     has_close_agent = False
        #     for agent in world.agents:
        #         if self.is_close(landmark, agent):
        #             has_close_agent = True
        #             break
        #     if not has_close_agent:
        #         return False
        # return True

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        # comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            # comm.append(other.state.c) # what is this? seems useless
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + entity_pos + other_pos)
                # [vx, vy, x, y, x1, y1, x2, y2, x3, y3, x1, y1, x2, y2]
