import numpy as np

class HopfieldNetwork:
    def __init__(self,
                 n: int=25,
                 method: str="hebb",
                 if_asynchronic: bool=True,
                 random_state: int=42
                ):
        super.__init__()
        self.n = n
        self.method = method if method == "hebb" or method == "oji" else "hebb"
        self.if_asynchronic = if_asynchronic
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)

        self.states = np.ones(self.n, dtype=int)
        self.weights = np.zeros((self.n, self.n), dtype=float)
        self.biases = np.zeros(self.n, dtype=int)

    def train(self, patterns: np.ndarray) -> None:
        if patterns.ndim != 2 or patterns.shape[1] != self.n:
            raise ValueError("Invalid patterns array size!")

        if not np.all(np.isin(patterns, [-1, 1])):
            raise ValueError("Invalid patterns structure - should contain only (-1, 1) values")

        self._hebb(patterns) if self.method == "hebb" else self._oji(patterns)

    def recall(self, pattern: np.ndarray, max_iter: int=1000) -> dict:
        pattern = np.asarray(pattern)

        if pattern.shape != (self.n, ):
            raise ValueError("Invalid pattern size!")

        if not np.all(np.isin(pattern, [-1, 1])):
            raise ValueError("Invalid pattern structure - should contain only (-1, 1) values")

        self.states = pattern.copy()

        visited = {self.states.tobytes(): 0}

        for i in range(1, max_iter + 1):
            if (self.step()):
                return {"Result": "stable", "Last iteration": i, "Output states": self.states.copy(), "Energy": self.get_energy()}

            states_key = self.states.tobytes()
            
            if not self.if_asynchronic and states_key in visited:
                return {"Result": "cycle detected in synchronic mode", "Last iteration": i, "Output states": self.states.copy(), "Energy": self.get_energy()}

            visited[states_key] = i

        return {"Result": "max iterations reached", "Last iteration": i, "Output states": self.states.copy(), "Energy": self.get_energy()}

    def step(self) -> bool:
        return self._async_step() if self.if_asynchronic else self._sync_step()

    def get_energy(self) -> float:
        return -0.5 * (self.states @ self.weights @ self.states) - (self.biases @ self.states)

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def get_states(self) -> np.ndarray:
        return self.states.copy()

    def _sync_step(self) -> bool:
        a = self.weights @ self.states + self.biases
        if_stable = True
        for i in range(self.n):
            new_state = self._sign_activate(a[i], i)
            if new_state != self.states[i]:
                if_stable = False
            self.states[i] = new_state
        return if_stable

    def _async_step(self) -> bool:
        if_stable = True
        for i in self.rng.permutation(self.n):
            a = self.weights[i] @ self.states + self.biases[i]
            new_state = self._sign_activate(a, i)
            if new_state != self.states[i]:
                if_stable = False
            self.states[i] = new_state
        return if_stable

    def _hebb(self, patterns: np.ndarray) -> None:
        for i in range(self.n):
            for j in range(self.n):
                self.weights[i][j] = np.sum(patterns[p][i] * patterns[p][j] for p in range(patterns.shape[0])) / self.n if i != j else 0

    def _oji(self, patterns: np.ndarray) -> None:
        pass

    def _sign_activate(self, a, i) -> int:
        if a < 0:
            return -1
        elif a > 0:
            return 1
        return self.states[i]