import numpy as np
import matplotlib.pyplot as plt
import math

class HopfieldNetwork:
    def __init__(self,
                 n: int=25,
                 method: str="hebb",
                 if_asynchronic: bool=True,
                 lr=0.01,
                 epochs=100,
                 random_state: int=42
                ):
        super().__init__()
        self.n = n
        self.method = method if method == "hebb" or method == "oji" else "hebb"
        self.if_asynchronic = if_asynchronic
        self.lr = lr
        self.epochs = epochs
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

        if pattern.shape[0] != self.n:
            raise ValueError("Invalid pattern size!")

        if not np.all(np.isin(pattern, [-1, 1])):
            raise ValueError("Invalid pattern structure - should contain only (-1, 1) values")

        self.states = pattern.copy()

        history = [self.states.copy()]
        visited = {self.states.tobytes(): 0}

        for i in range(1, max_iter + 1):
            is_stable = self.step()
            history.append(self.states.copy())

            if (is_stable):
                return {"Result": "stable", "Last iteration": i, "Output states": self.states.copy(), "Energy": self.get_energy(), "History": history}

            states_key = self.states.tobytes()
            
            if not self.if_asynchronic and states_key in visited:
                return {"Result": "cycle detected in synchronic mode", "Last iteration": i, "Output states": self.states.copy(), "Energy": self.get_energy(), "History": history}

            visited[states_key] = i

        return {"Result": "max iterations reached", "Last iteration": i, "Output states": self.states.copy(), "Energy": self.get_energy(), "History": history}

    def plot_history(self, history: list, plot_dim: tuple) -> None:
        if math.prod(plot_dim) != self.n:
            raise ValueError("Incorrect bitmap's dimensions")

        count = len(history)
        cols = min(5, count)
        rows = math.ceil(count / cols)

        _, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8), squeeze=False)

        for k, ax in enumerate(axes.flat):
            if k < count:
                bitmap = history[k].reshape(plot_dim)

                ax.imshow(bitmap, cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest")
                ax.set_title(f"Iteration: {k}")

            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def step(self) -> bool:
        return self._async_step() if self.if_asynchronic else self._sync_step()

    def get_energy(self) -> float:
        return -0.5 * (self.states @ self.weights @ self.states) - (self.biases @ self.states)

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def get_states(self) -> np.ndarray:
        return self.states.copy()

    def change_method(self, method: str="hebb"):
        if method == "hebb" or method == "oji":
            self.method = method

    def change_sync_mode(self, if_asynchronic: bool=True):
        self.if_asynchronic = if_asynchronic

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
        self.weights = (patterns.T @ patterns) / self.n
        np.fill_diagonal(self.weights, 0)

    def _oji(self, patterns: np.ndarray) -> None:
        self.weights = self.rng.normal(loc=0.0, scale=0.01, size=(self.n, self.n))
        self.weights = (self.weights + self.weights.T) / 2
        np.fill_diagonal(self.weights, 0)
        
        for _ in range(self.epochs):
            for p in patterns:
                a = self.weights @ p
                for i in range(self.n):
                    for j in range(self.n):
                        self.weights[i][j] = self.weights[i][j] + self.lr * a[i] * (p[j] - a[i] * self.weights[i][j])
                self.weights = (self.weights + self.weights.T) / 2
                np.fill_diagonal(self.weights, 0)

    def _sign_activate(self, a, i) -> int:
        if a < 0:
            return -1
        elif a > 0:
            return 1
        return self.states[i]