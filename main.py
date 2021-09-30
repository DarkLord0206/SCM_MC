import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from scm_env import ScmEnv


class MC:
    def __init__(self, input_dims, output_dims):
        self.env = ScmEnv()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.epsilon = 1000
        self.X = layers.Input(shape=(input_dims,))
        net = self.X
        net = layers.Dense(10)(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(output_dims)(net)

        self.model = Model(inputs=self.X, outputs=net)

        new_probs_placeholder = K.placeholder(shape=(None,), name="new_probs")
        probs = self.model.output
        loss = (new_probs_placeholder - probs) ** 2 * 0.5

        adam = Adam(learning_rate=0.001)

        updates = adam.get_updates(params=self.model.trainable_weights, loss=loss)

        self.train_fn = K.function(inputs=[self.model.input, new_probs_placeholder], outputs=[self.model.outputs],
                                   updates=updates)

    def fit(self, state_list, action_list, reward_list):
        rewards = {}
        actions = {}
        unique_states = []
        for x, y, z in zip(state_list, action_list, reward_list):
            k = x.copy()
            x = x.tobytes()
            if x in actions:
                if y not in actions[x]:
                    actions[x].append(y)
            else:
                unique_states.append(k)
                actions[x] = [y]
            if (x, y) in rewards:
                rewards[(x, y)] += z
            else:
                rewards[(x, y)] = z
        for x, y in rewards:
            rewards[(x, y)] /= len(actions[x])
        new_probs_list = []
        a_stars = []

        for x in actions:
            maxr = 0
            maxv = -1
            for y in actions[x]:
                if rewards[(x, y)] > maxv:
                    maxv = rewards[(x, y)]
                    maxr = y
            a_stars.append(maxr)
        new_probs = [self.epsilon / self.output_dims] * self.output_dims
        for x in a_stars:
            temp = new_probs.copy()
            temp[x] = 1 - self.epsilon + self.epsilon / self.output_dims
            new_probs_list.append(temp)
        self.train_fn(unique_states, new_probs_list)

    def computer_action(self, state):
        probs = self.model.predict(state)
        action = np.random.choice(self.output_dims, p=probs)
        return action

    def create_episode(self):
        state = self.env.reset()
        done = False
        state_list = []
        action_list = []
        reward_list = []
        while not done:
            action = self.computer_action(state)
            new_state, reward, done, info = self.env.step(action)
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            state = new_state
        total_reward = sum(reward_list)
        print(total_reward)
        self.fit(state_list, action_list, reward_list)
