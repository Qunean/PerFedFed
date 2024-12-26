import numpy as np
import torch
from sklearn import metrics


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


class Metrics:
    def __init__(self, loss=None, predicts=None, targets=None):
        self._loss = loss if loss is not None else 0.0
        self._targets = targets if targets is not None else []
        self._predicts = predicts if predicts is not None else []

    def update(self, other):
        if other is not None:
            self._predicts.extend(to_numpy(other._predicts))
            self._targets.extend(to_numpy(other._targets))
            self._loss += other._loss

    def _calculate(self, metric, **kwargs):
        return metric(self._targets, self._predicts, **kwargs)

    @property
    def loss(self):
        if len(self._targets) > 0:
            return self._loss / len(self._targets)
        else:
            return 0.0

    @property
    def macro_precision(self):
        score = self._calculate(
            metrics.precision_score, average="macro", zero_division=0
        )
        return score * 100

    @property
    def macro_recall(self):
        score = self._calculate(metrics.recall_score, average="macro", zero_division=0)
        return score * 100

    @property
    def micro_precision(self):
        score = self._calculate(
            metrics.precision_score, average="micro", zero_division=0
        )
        return score * 100

    @property
    def micro_recall(self):
        score = self._calculate(metrics.recall_score, average="micro", zero_division=0)
        return score * 100

    @property
    def accuracy(self):
        if self.size == 0:
            return 0
        score = self._calculate(metrics.accuracy_score)
        return score * 100

    @property
    def corrects(self):
        return self._calculate(metrics.accuracy_score, normalize=False)

    @property
    def size(self):
        return len(self._targets)

class ASRMetrics:
    """
    专门用于记录和计算攻击成功率（ASR: Attack Success Rate）的类
    """
    def __init__(self):
        self._correct = 0  # 成功触发并预测到目标类别的样本数
        self._total = 0    # 总触发样本数

    def update(self, correct: int = None, total: int = None, other: "ASRMetrics" = None):
        # print(f"update called with correct={correct}, total={total}, other={type(other)}")
        if other is not None:
            if not isinstance(other, ASRMetrics):
                raise TypeError(f"The 'other' parameter must be an instance of ASRMetrics, got {type(other)}.")
            self._correct += other._correct
            self._total += other._total
        elif correct is not None and total is not None:
            if total < 0 or correct < 0:
                raise ValueError("Both 'correct' and 'total' must be non-negative integers.")
            self._correct += correct
            self._total += total
        else:
            raise ValueError("Either 'other' (ASRMetrics instance) or both 'correct' and 'total' must be provided.")

    def reset(self):
        """
        重置统计数据
        """
        self._correct = 0
        self._total = 0

    @property
    def asr(self) -> float:
        """
        计算攻击成功率 (ASR)
        :return: 攻击成功率的百分比 (0-100%)
        """
        if self._total == 0:
            return 0.0
        return (self._correct / self._total) * 100

    @property
    def correct(self) -> int:
        """
        返回成功攻击的样本数
        """
        return self._correct

    @property
    def total(self) -> int:
        """
        返回触发样本的总数量
        """
        return self._total

    def __str__(self):
        return f"ASR: {self.asr:.2f}%, Correct: {self._correct}, Total: {self._total}"
