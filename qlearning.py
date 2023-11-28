from sgym.games._2048 import Environment
import random
from collections import deque

LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
START_EPSILON = 0.9
FINAL_EPSILON = 0.05
EPSILON = START_EPSILON
MAX_STEPS = 5000


class QLearningAgent:
    def __init__(self) -> None:
        self.q_table: dict = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, value):
        self.q_table[(state, action)] = value

    def choose_action(self, state, actions, epsilon=EPSILON):
        if random.random() < epsilon:
            return random.choice(actions)
        else:
            q_values = [self.get_q_value(state, action) for action in actions]
            max_q = max(q_values)
            count = q_values.count(max_q)
            if count > 1:
                best = [i for i in range(len(actions)) if q_values[i] == max_q]
                i = random.choice(best)
            else:
                i = q_values.index(max_q)
            return actions[i]

    def learn(self, env, max_episodes=1000, max_steps=MAX_STEPS):
        best_score = 0
        epsilon = START_EPSILON
        all_scores = deque(maxlen=1000)
        for episode in range(max_episodes):
            state = env.reset()
            total_reward = 0.0
            epsilon = max(START_EPSILON * (1 - episode / max_episodes), FINAL_EPSILON)
            for step in range(max_steps):
                actions = env.get_actions()
                action = self.choose_action(state, actions, epsilon=epsilon)
                next_state, reward, done, _, info = env.step(action)
                total_reward += reward
                q_value = self.get_q_value(state, action)
                best = max([self.get_q_value(next_state, a) for a in actions])
                new_q = (1 - LEARNING_RATE) * q_value + LEARNING_RATE * (
                    reward + DISCOUNT_FACTOR * best
                )
                self.update_q_value(state, action, new_q)
                state = next_state
                if done:
                    if info["score"] > best_score:
                        best_score = info["score"]
                        print(
                            f"New best score: {best_score}, current epsilon: {epsilon*100:.2f} at episode {episode}"
                        )
                        if best_score >= 4000:
                            self.play(env, epsilon=epsilon)
                    break

            # print(f"Episode {episode} finished after {step + 1} steps with total reward {total_reward:.2f}")
            all_scores.append(info["score"])
            if episode % 1000 == 0:
                print(
                    f"Episode {episode}: "
                    f"avg. score over last 1000 episodes: {sum(all_scores)/1000:.2f}"
                )

    def play(self, env, max_steps=MAX_STEPS, epsilon=0):
        state = env.reset()
        env.render = True
        total_reward = 0.0
        for step in range(max_steps):
            actions = env.get_actions()
            action = self.choose_action(state, actions, epsilon=epsilon)
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        print(f"Game finished after {step + 1} steps with score: {info['score']:.2f}")
        env.render = False


def main():
    env = Environment(render=False)
    agent = QLearningAgent()
    agent.learn(env, max_episodes=50000)


if __name__ == "__main__":
    main()
