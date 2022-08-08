import gym
from metaworld import MT50
from metaworld.policies import *
from typing import Optional
import random

mt = MT50(seed=0)


MT10_TASK = [ 'drawer-close-v2', 'reach-v2', 'window-close-v2', 'window-open-v2', 'button-press-topdown-v2',
              'door-open-v2', 'drawer-open-v2', 'pick-place-v2', 'peg-insert-side-v2', 'push-v2']

MT10_EXPERT = {'drawer-close-v2':SawyerDrawerCloseV2Policy, 'reach-v2':SawyerReachV2Policy, 'window-close-v2': SawyerWindowCloseV2Policy,
                'window-open-v2':SawyerWindowOpenV2Policy, 'button-press-topdown-v2': SawyerButtonPressTopdownV2Policy,
                'door-open-v2':SawyerDoorOpenV2Policy, 'drawer-open-v2': SawyerDrawerOpenV2Policy, 'pick-place-v2': SawyerPickPlaceV2Policy,
                'peg-insert-side-v2':SawyerPegInsertionSideV2Policy, 'push-v2':SawyerPushV2Policy}


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
        if info['success']:
            done = True
            info['is_success'] = True

        if done and not info['success']:
            info['is_success'] = False

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


class GoalEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, expert, tasks):
        super().__init__(TimeLimitMDP(env))
        self._observation_space = gym.spaces.Dict(
            {
                "observations": self.env.observation_space,
                "goal": self.env.observation_space
            }
        )
        self.expert = expert
        self.goals = []
        self.tasks = tasks
        self.build_all_goal(self.tasks)
        assert len(self.tasks) == len(self.goals)
        self.goal = self.goals[0]

    def reset(self, **kwargs):
        task_index = random.randrange(0, len(self.tasks))
        self.goal = self.goals[task_index]
        return gym.ObservationWrapper.reset(self)

    def _build_goal(self):
        env = self.env
        obs = env.reset()
        success = False
        while not success:
            done = False
            while not done:
                action = self.expert.get_action(obs)
                obs, reward, done, info = env.step(action)
                success = info['success']
                if success:
                    done = True
                    break

        return obs

    def build_all_goal(self, tasks):
        from tqdm import tqdm
        for task_index, task in tqdm(enumerate(tasks)):
            self.env.set_task(task)
            self.goals.append(self._build_goal())

    def observation(self, observation):
        return {"observations": observation, "goal": self.goal}


def generate_env(domain_name: str, task_index: Optional[int] = None):
    tasks = [task for task in mt.train_tasks if task.env_name == domain_name]
    if task_index is None:
        task_index = random.randrange(0, len(tasks))
    policy = MT10_EXPERT[domain_name]()
    env_cls = mt.train_classes[domain_name]
    env = env_cls()
    env.set_task(tasks[task_index])
    test_env = GoalEnvWrapper(env, policy, tasks)
    return test_env


if __name__ == '__main__':
    generate_env("door-open-v2")