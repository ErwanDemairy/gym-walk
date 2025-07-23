import gymnasium as gym
import gym_walk, numpy as np
env = gym.make('WalkFive-v0')
pi = lambda x: np.random.randint(2)

def td(pi, env, gamma=1.0, alpha=0.01, n_episodes=100000):
    V = np.zeros(env.observation_space.n)
    for t in range(n_episodes):
        print(f"env.reset = {env.reset()}")
        state, state_p = env.reset()
        done = False
            
        while not done:
            print(f"state = {state}")
            action = pi(state)
#            next_state, reward, done, _ = env.step(action)
            next_state, reward, done, _, _ = env.step(action)
            print(f"next_state = {next_state}")
            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            V[state] = V[state] + alpha * td_error
            state = next_state
    return V

V = td(pi, env)
V
