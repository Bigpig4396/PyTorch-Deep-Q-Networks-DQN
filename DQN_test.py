from env_FindGoals import EnvFindGoals
from DQN import Agent

if __name__ == '__main__':
    env = EnvFindGoals()
    max_iter = 2000
    agent = Agent(5)

    # training phase
    for i in range(max_iter):
        print("iter= ", i)
        obs = env.get_agt1_obs()
        action = agent.get_action(obs, 0.8)
        action_list = [action, 4]
        reward_list, done = env.step(action_list)
        next_obs = env.get_agt1_obs()
        agent.remember(obs, action, reward_list[0], done, next_obs)
        agent.train()

    # test phase
    env.reset()
    for i in range(max_iter):
        print("iter= ", i)
        obs = env.get_agt1_obs()
        action = agent.get_action(obs, 0)
        action_list = [action, 4]
        env.plot_scene()
        reward_list, done = env.step(action_list)
