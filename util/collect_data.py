import numpy as np
import tqdm
from metaworld.policies import *
import gym
from metaworld import MT50


MT10_TASK = ['pick-place-v2', 'reach-v2', 'window-close-v2', 'window-open-v2', 'button-press-topdown-v2',
             'door-open-v2', 'drawer-open-v2', 'pick-place-v2','drawer-close-v2', 'peg-insert-side-v2', 'push-v2']

MT10_EXPERT = {
        'drawer-close-v2': SawyerDrawerCloseV2Policy,
        'reach-v2': SawyerReachV2Policy,
        'window-close-v2': SawyerWindowCloseV2Policy,
        'window-open-v2': SawyerWindowOpenV2Policy,
        'button-press-topdown-v2': SawyerButtonPressTopdownV2Policy,
        'door-open-v2': SawyerDoorOpenV2Policy,
        'drawer-open-v2': SawyerDrawerOpenV2Policy,
        'pick-place-v2': SawyerPickPlaceV2Policy,
        'peg-insert-side-v2': SawyerPegInsertionSideV2Policy,
        'push-v2': SawyerPushV2Policy
    }


class TimeLimitMDP(gym.Wrapper):
    def __init__(self, env, max_episode_steps=200):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.success = False
        self.reward = 0

        self.metadata = {'render_modes': ['human', 'rgb_array']}

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True

        return state, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self.success = False
        self.reward = 0
        return self.env.reset(**kwargs)

    def render(self):
        out = self.env.render(offscreen=False)
        # cv2.imwrite('tmp.jpeg', out)
        return out


class TimeLimitRewardMDP(TimeLimitMDP):
    def step(self, action):
        state, reward, done, info = super().step(action)
        info['reward'] = reward
        shaping_reward = reward - self.reward
        self.reward = reward

        # if info['success']:
        #     done = True

        return state, shaping_reward, done, info


MIN_LEN = 50
SEED = 0
NOISE = 0.25

if __name__ == "__main__":
    datasets = []
    mt = MT50(seed=SEED)
    for domain_name in MT10_TASK:
        # Set environments
        env_cls = mt.train_classes[domain_name]
        env = env_cls()

        # Set rule based expert policy
        policy = MT10_EXPERT[domain_name]()

        # Prepare task dataset
        domain_dataset = []
        goal_dataset = {}
        n_transitions = 0

        tasks = [task for task in mt.train_tasks if task.env_name == domain_name]

        task_num = 0
        dictionary = {"observation": [],
                      "action": [],
                      "reward": [],
                      "terminals": [],
                      "goal": [],
                      "task_index": []}
        episodes = []
        for task_idx, task in enumerate(tasks):
            env.set_task(task)
            test_env = TimeLimitMDP(env)
            progress_bar = tqdm.tqdm(range(250))
            for epochs in progress_bar:
                observations = []
                next_observations = []
                images = []
                next_images = []
                actions = []
                rewards = []
                terminals = []
                infos = []

                ep_len = 0
                success = False
                while not success:
                    observation = test_env.reset()
                    done = False
                    while not done:
                        action = policy.get_action(observation) + (np.random.uniform(size=4) - 0.5) * NOISE
                        action = np.clip(action, -1.0, 1.0)
                        next_observation, reward, done, info = test_env.step(action)
                        observations.append(observation.copy())
                        next_observations.append(next_observation.copy())
                        actions.append(action)
                        rewards.append(reward)
                        info.update(
                            {
                                "domain_name": domain_name,
                                "task_idx": task_idx
                            }
                        )
                        if info["success"]:
                            goal_dataset[task_idx] = {
                                "goal_observation": next_observation,
                            }
                            done = True
                            goal = next_observation.copy()
                            success = True
                        observation = next_observation
                        terminals.append(done)
                        ep_len += 1

                ones_like = np.ones_like(observations)
                goal = np.asarray(goal, dtype=np.float32)[None]
                goal = goal * ones_like

                task_index = np.ones(shape=(len(observations)), dtype=np.int32) * task_idx
                one_episodes = {
                    "observations": np.asarray(observations, dtype=np.float32),
                    "next_observations": np.asarray(next_observations, dtype=np.float32),
                    "actions": np.asarray(actions, dtype=np.float32),
                    "rewards": np.asarray(rewards, dtype=np.float32),
                    "terminals": np.asarray(terminals, dtype=np.float32),
                    "goal": goal,
                    "task_index": task_index
                }
                episodes.append(one_episodes)

        # list[dict] to dict[array]
        data_buffer = {k: [] for k in episodes[0].keys()}
        data_buffer["epilen"] = []
        data_buffer["time_step"] = []
        data_buffer["episode_id"] = []

        for i, epi in enumerate(episodes):
            for k in epi.keys():
                data_buffer[k].append(epi[k])
            epilen = len(epi['actions'])
            data_buffer["epilen"].append(np.ones(epilen, dtype=np.int32) * epilen)
            data_buffer["time_step"].append(np.arange(epilen, dtype=np.int32))
            data_buffer['episode_id'].append(np.ones(epilen, dtype=np.int32) * i)

        for k in data_buffer.keys():
            data_buffer[k] = np.concatenate(data_buffer[k], axis=0)
        np.savez("data/{}".format(domain_name), **data_buffer)
