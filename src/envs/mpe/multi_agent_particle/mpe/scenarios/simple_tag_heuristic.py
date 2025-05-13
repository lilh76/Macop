import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario

scalar = 1

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 5 # prey
        num_adversaries = 2 # predator
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 1.5
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 0.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.0
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        count_prey = 0
        for agent in world.agents:
            if agent.adversary:
                agent.state.p_pos = np.random.uniform(-0.1, +0.1, world.dim_p)

            else: # prey
                if count_prey == 0:
                    agent.state.p_pos = np.array([1., 0.]) * scalar
                elif count_prey == 1:
                    agent.state.p_pos = np.array([0.31, 0.95]) * scalar
                elif count_prey == 2:
                    agent.state.p_pos = np.array([-0.8, 0.6]) * scalar
                elif count_prey == 3:
                    agent.state.p_pos = np.array([-0.8, -0.6]) * scalar
                elif count_prey == 4:
                    agent.state.p_pos = np.array([0.31, -0.95]) * scalar
                count_prey += 1

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.array([5, 0])

        self.is_done = False

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min * 2 else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        rew = 0
        # if abs(agent.state.p_pos).max() > 1:
        #     rew -= 0.01
        if self.done(agent, world) and not self.is_done:
            self.is_done = True
            rew += 1.0
        return rew

    def done(self, agent, world):
        # done if one prey is caught
        all_preys = self.good_agents(world)
        for prey in all_preys:
            if sum([self.is_collision(prey, adv) for adv in self.adversaries(world)]) >= 2:
                return True
        return False

    # def agent_reward(self, agent, world):
    #     return 0.0

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary: # if other is prey
                other_vel.append(other.state.p_vel)
        # return np.concatenate(entity_pos + [agent.state.p_pos] + [agent.state.p_vel] + other_pos + other_vel)
        return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + other_pos + other_vel) # len=22 for predator, len=20 for prey
