from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.special import softmax
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def align_probabilities(probabilities, classes):
    result = np.full((len(probabilities), 3), 1e-12)
    result[:, np.asarray(classes, dtype=int)] = probabilities
    return result / result.sum(1, keepdims=True)


class ClassicalModel:
    def __init__(self, name, seed, iterations):
        self.name, self.seed, self.iterations = name, seed, iterations
        self.stats = {"optimizer_iterations": 0, "objective_evaluations": None,
                      "circuit_executions": 0, "measurement_shots": 0}

    def fit(self, x, y, deadline=None):
        classes = np.unique(y)
        if self.name == "majority" or len(classes) == 1:
            self.model = DummyClassifier(strategy="most_frequent").fit(x, y)
            self.stats["fallback"] = "single_class" if len(classes) == 1 else None
        elif self.name == "logistic":
            self.model = LogisticRegression(max_iter=self.iterations, random_state=self.seed).fit(x, y)
            self.stats["optimizer_iterations"] = int(self.model.n_iter_.max())
        elif self.name == "xgboost":
            from xgboost import XGBClassifier
            # XGBoost expects dense class IDs even when a chronological window lacks a class.
            self.class_mapping = classes
            encoded = np.searchsorted(classes, y)
            self.model = XGBClassifier(n_estimators=self.iterations, max_depth=3, learning_rate=0.1,
                                      n_jobs=1, random_state=self.seed, tree_method="hist")
            self.model.fit(x, encoded)
            self.stats["optimizer_iterations"] = self.iterations
        elif self.name == "mlp":
            import torch
            torch.set_num_threads(1)
            torch.manual_seed(self.seed)
            self.model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 16), torch.nn.Tanh(),
                                             torch.nn.Linear(16, 3)).double()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.025, weight_decay=1e-3)
            inputs, targets = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.long)
            for step in range(self.iterations):
                if deadline is not None and perf_counter() >= deadline:
                    raise TimeoutError("MLP training exceeded fit budget")
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(inputs), targets)
                loss.backward()
                optimizer.step()
                self.stats["optimizer_iterations"] = step + 1
            self.stats["objective_evaluations"] = self.stats["optimizer_iterations"]
            self.model.eval()
        else:
            raise ValueError(f"unknown classical model {self.name}")
        return self

    def predict_proba(self, x):
        if hasattr(self.model, "predict_proba"):
            mapping = getattr(self, "class_mapping", self.model.classes_)
            return align_probabilities(self.model.predict_proba(x), mapping)
        import torch
        with torch.no_grad():
            logits = self.model(torch.as_tensor(x, dtype=torch.float64)).numpy()
        return softmax(logits, axis=1)
