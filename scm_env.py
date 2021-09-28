import numpy as np

rng = np.random.default_rng()


class scmEnv:
    def __init__(self):
        self.periods = 30
        self.inv = np.array([100, 100, 200])  # Initial Inventory
        self.unit_price = np.array([2, 1.5, 1, 0.75])  # unit sales price
        self.unit_cost = np.array([1.5, 1.0, 0.75, 0.5])  # unit replishment cost
        self.demand_cost = np.array([0.10, 0.075, 0.05, 0.025])  # unit backlog cost
        self.holding_cost = np.array([0.15, 0.10, 0.05, 0])  # unit holding cost
        self.supply_capacity = np.array([50, 30, 20])  # production capacity
        self.lead_time = np.array([3, 5, 10])  # lead times
        self.backlog = False
        self.user_D = np.zeros(self.periods)
        self.max_rewards = 2000
        self.discount = 0.9
        self.map = np.array([[a, b, c] for a in range(50) for b in range(30) for c in range(20)])
        lt_max = self.lead_time.max()
        self.action_space_len = 50 * 30820
        self.rvs = np.random.default_rng().poisson(100, self.periods)
        self.obervation_space = []
        self.observation_space_len = 4 * (lt_max + 1)

    def _reset(self):
        self.I = np.zeros([self.periods + 1, 3])
        self.T = np.zeros([self.periods + 1, 3])
        self.R = np.zeros([self.periods, 3])
        self.D = np.zeros(30)
        self.sales = np.zeros([30, 4])
        self.profit = np.zeros(self.periods)
        self.period = 0
        self.I[0, :] = self.inv
        self.action_log = np.zeros((30, 3))

        self._update_state()

        return self.state

    def _update_state(self):
        m = 3
        lt_max = self.lead_time.max()
        t = self.period
        state = np.zeros(3 * (lt_max + 1))
        lt_max = self.lead_time.max()
        if t == 0:
            state[:m] = self.inv
        else:
            state[:m] = self.I[t]
        if t == 0:
            pass
        elif t >= lt_max:
            state[-m * lt_max:] += self.action_log[t - lt_max:t].flatten()
        else:
            state[-m * t:] += self.action_log[:t].flatten()
        self.state = state.copy()

    def mapper(self, action):
        return self.map[action]

    def step(self, action):
        action = self.mapper(action)
        R = np.maximum(action, 0).astype(int)
        n = self.period
        L = self.lead_time
        I = self.I[n, :].copy()
        T = self.T[n, :].copy()
        m = 4

        c = self.supply_capacity
        self.action_log[n] = R.copy
        Im1 = np.append(I[1:], np.inf)

        Rcopy = R.copy()
        R[R >= c] = c[R >= c]
        R[R >= Im1] = Im1[R >= Im1]
        self.R[n, :] = R
        RnL = np.zeros(m - 1)
        for i in range(m - 1):
            RnL[i] = self.R[n - L[i], i].copy()
            I[i] = I[i] + RnL[i]
        self.D[n] = self.rvs[n]
        D = self.rvs[n]
        S0 = min(I[0], 0)
        S = np.append(S0, R)
        self.sales[n, :] = S
        I = I - S[:-1]
        T = T - RnL + R
        self.I[n + 1, :] = I
        self.T[n + 1, :] = T
        U = np.append(D, Rcopy) - S
        LS = U

        p = self.unit_price
        r = self.unit_cost
        k = self.demand_cost
        h = self.holding_cost
        a = self.discount
        II = np.append( I, 0)
        RR = np.append(R, S[-1])
        P = a ** n * np.sum(p * S - (r * RR + k * U + h * II))
        self.profit[n] = P

        self.period += 1

        self._update_state()

        reward = P

        if self.period >= 30:
            done = True
        else:
            done = False
        return self.state, reward, done, {}
