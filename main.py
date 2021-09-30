import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from scm_env import ScmEnv

tf.compat.v1.disable_eager_execution()


class MC:
    def __init__(self):
        self.env = ScmEnv()  # creating the envoirement
        self.input_dims = self.env.observation_space_len
        self.output_dims = self.env.action_space_len
        self.epsilon = 1000  # to add exploration
        self.X = layers.Input(shape=(self.input_dims,))  # defining the NN for the agent
        net = self.X
        net = layers.Dense(10)(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(self.output_dims)(net)
        net = layers.Activation("softmax")(net)

        self.model = Model(inputs=self.X, outputs=net)

        new_probs_placeholder = K.placeholder(shape=(None,), name="new_probs")
        probs = self.model.output
        loss = (new_probs_placeholder - probs) ** 2 * 0.5  # defining the loss function

        adam = Adam(learning_rate=0.001)  # adam optimizer

        updates = adam.get_updates(params=self.model.trainable_weights, loss=loss)

        self.train_fn = K.function(inputs=[self.model.input, new_probs_placeholder], outputs=[self.model.outputs],
                                   updates=updates)  # custom train function

    def fit(self, state_list, action_list, reward_list):
        rewards = {}
        actions = {}
        unique_states = []
        for x, y, z in zip(state_list, action_list, reward_list):
            k = x.copy()
            x = x.tobytes()  # numoy array isnt hashable so has to be converted to bytes
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
        for x, y in rewards: # Monte Carlo method
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
            new_probs_list.append(temp) # storing the new probs
        new_probs_list = np.asarray(new_probs_list)
        unique_states = np.asarray(unique_states)
        self.train_fn([unique_states, new_probs_list])

    def computer_action(self, state):
        probs = self.model.predict(np.asarray([state]))
        action = np.random.choice(self.output_dims, p=probs[0])
        return action # select action according

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


agent = MC()
for _ in range(1000):
    agent.create_episode()
