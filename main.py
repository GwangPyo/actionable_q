from actionable_sac.actionable_sac import ActionableSAC
from actionable_sac.policy import GoalConditionMlpPolicy
from dataset.offline_dataset import ActionableDataset
from util.env_wrapper import generate_env
import numpy as np
from tqdm import tqdm
from util.keras_progbar import Progbar
import argparse


def evaluate(env, model, steps=100):
    scores = []
    successes = []
    for _ in tqdm(range(steps)):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                scores.append(score)
                successes.append(info['success'])
    print("eval")
    print("reward", np.mean(scores) , "+/-", np.std(scores))
    print("success", 100 * np.mean(successes), "%")
    return np.mean(scores), np.mean(successes)


def main(env_name, evaluate_interval: int, num_eval: int = 100):
    env = generate_env(env_name)
    model = ActionableSAC(env=env, policy=GoalConditionMlpPolicy, verbose=1,)
    model._setup_learn(100000, eval_env=env)
    dataset = ActionableDataset(f'data/{env_name}.npz')

    for _ in range(1000):
        progbar = Progbar(evaluate_interval)
        for i in range(evaluate_interval):
            progbar.add(1, model.offline_train_step(dataset.fetch()))
        evaluate(env, model, num_eval)
    model.save(f"{env_name}-v2")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_name", default='drawer-close-v2', type=str)
    parser.add_argument('-n', "--num_eval", default=100, type=int)
    parser.add_argument('-i', "--eval_interval", default=500, type=int)
    args = parser.parse_args()
    main(args.env_name, args.eval_interval, args.num_eval)
