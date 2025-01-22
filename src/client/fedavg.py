from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
from omegaconf import DictConfig
from ray.tune.examples.pbt_dcgan_mnist.common import batch_size
from torch.utils.data import DataLoader, Subset

from data.utils.datasets import BaseDataset
from src.utils.metrics import Metrics,ASRMetrics
from src.utils.models import DecoupledModel
from src.utils.tools import evalutate_model, get_optimal_cuda_device,evaluate_asr_model,save_tensor_as_image,add_trigger


class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        optimizer_cls: type[torch.optim.Optimizer],
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler],
        args: DictConfig,
        dataset: BaseDataset,
        data_indices: list,
        device: torch.device | None,
        return_diff: bool,
    ):
        self.client_id: int = None
        self.args = args
        if device is None:
            self.device = get_optimal_cuda_device(use_cuda=self.args.common.use_cuda)
        else:
            self.device = device
        self.dataset = dataset
        self.model = model.to(self.device)
        self.regular_model_params: OrderedDict[str, torch.Tensor]
        self.personal_params_name: list[str] = []
        self.regular_params_name = list(key for key, _ in self.model.named_parameters())
        if self.args.common.buffers == "local":
            self.personal_params_name.extend(
                [name for name, _ in self.model.named_buffers()]
            )
        elif self.args.common.buffers == "drop":
            self.init_buffers = deepcopy(OrderedDict(self.model.named_buffers()))

        self.optimizer = optimizer_cls(params=self.model.parameters())
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.init_lr_scheduler_state: dict = None
        self.lr_scheduler_cls = None
        if lr_scheduler_cls is not None:
            self.lr_scheduler_cls = lr_scheduler_cls
            self.lr_scheduler = self.lr_scheduler_cls(optimizer=self.optimizer)
            self.init_lr_scheduler_state = deepcopy(self.lr_scheduler.state_dict())

        # [{"train": [...], "val": [...], "test": [...]}, ...]
        self.data_indices = data_indices
        # Please don't bother with the [0], which is only for avoiding raising runtime error by setting Subset(indices=[]) with `DataLoader(shuffle=True)`
        self.trainset = Subset(self.dataset, indices=[0])
        self.valset = Subset(self.dataset, indices=[])
        self.testset = Subset(self.dataset, indices=[])
        self.trainloader = DataLoader(
            self.trainset, batch_size=self.args.common.batch_size, shuffle=True
        )
        self.valloader = DataLoader(self.valset, batch_size=self.args.common.batch_size)
        self.testloader = DataLoader(
            self.testset, batch_size=self.args.common.batch_size
        )
        self.testing = False

        self.local_epoch = self.args.common.local_epoch
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.eval_results = {}

        self.return_diff = return_diff
        self.malicious = False
        self.server_current_epoch = None


    def load_data_indices(self):
        """This function is for loading data indices for No.`self.client_id`
        client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def train_with_eval(self):
        """Wraps `fit()` with `evaluate()` and collect model evaluation
        results.

        A model evaluation results dict: {
                `before`: {...}
                `after`: {...}
                `message`: "..."
            }
            `before` means pre-local-training.
            `after` means post-local-training
        """
        eval_results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        if self.args.common.malicious_ratio>0:
            eval_results["malicious"] = {"train": ASRMetrics(), "val": ASRMetrics(), "test": ASRMetrics()}
        eval_results["before"] = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            eval_results["after"] = self.evaluate()
            if self.args.common.malicious_ratio >0 :
                eval_results["malicious"] = self.asr_test()

        eval_msg = []
        for split, color, flag, subset in [
            ["train", "yellow", self.args.common.eval_train, self.trainset],
            ["val", "green", self.args.common.eval_val, self.valset],
            ["test", "cyan", self.args.common.eval_test, self.testset],
        ]:
            if len(subset) > 0 and flag:
                eval_msg.append(
                    f"client [{self.client_id}]\t"
                    f"[{color}]({split}set)\t"
                    f"loss: {eval_results['before'][split].loss:.4f} -> {eval_results['after'][split].loss:.4f}\t"
                    f"accuracy: {eval_results['before'][split].accuracy:.2f}% -> {eval_results['after'][split].accuracy:.2f}%"
                )
                # 添加 malicious 信息
                if self.args.common.malicious_ratio > 0 and "malicious" in eval_results:
                    malicious_results = eval_results["malicious"].get(split, None)
                    if malicious_results:
                        # 根据 self.malicious 设置颜色
                        color = "red" if self.malicious else "black"
                        eval_msg.append(
                            f"client [{self.client_id}]\t"
                            f"[{color}]({split}set-malicious)\t"
                            f"ASR: {malicious_results.asr:.2f}%\t"
                            f"Correct: {malicious_results.correct}/{malicious_results.total}"
                        )

        eval_results["message"] = eval_msg
        self.eval_results = eval_results

    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        self.model.load_state_dict(package["regular_model_params"], strict=False)
        self.model.load_state_dict(package["personal_model_params"], strict=False)
        if self.args.common.buffers == "drop":
            self.model.load_state_dict(self.init_buffers, strict=False)

        if self.return_diff:
            model_params = self.model.state_dict()
            self.regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )
        # ------ malicious---
        self.malicious = True if package["malicious"] == 1 else False
        self.server_current_epoch = package["current_epoch"]

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.train_with_eval()
        client_package = self.package()
        return client_package

    def package(self):
        """Package data that client needs to transmit to the server. You can
        override this function and add more parameters.

        Returns:
            A dict: {
                `weight`: Client weight. Defaults to the size of client training set.
                `regular_model_params`: Client model parameters that will join parameter aggregation.
                `model_params_diff`: The parameter difference between the client trained and the global. `diff = global - trained`.
                `eval_results`: Client model evaluation results.
                `personal_model_params`: Client model parameters that absent to parameter aggregation.
                `optimzier_state`: Client optimizer's state dict.
                `lr_scheduler_state`: Client learning rate scheduler's state dict.
            }
        """
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].clone().cpu() for key in self.regular_params_name
            },
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
        )
        if self.return_diff:
            client_package["model_params_diff"] = {
                key: param_old - param_new
                for (key, param_new), param_old in zip(
                    client_package["regular_model_params"].items(),
                    self.regular_model_params.values(),
                )
            }
            client_package.pop("regular_model_params")
        return client_package

    # def fit(self):
    #     self.model.train()
    #     self.dataset.train()
    #     if self.malicious == True:
    #         attack(self.trainloader,self.args.common.attack_method)
    #     for _ in range(self.local_epoch):
    #         for x, y in self.trainloader:
    #             # 当当前批次大小为1时，BatchNorm2d会报错，跳过该批次
    #             if len(x) <= 1:
    #                 continue
    #
    #             x, y = x.to(self.device), y.to(self.device)
    #             logit = self.model(x)
    #             loss = self.criterion(logit, y)
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #
    #         if self.lr_scheduler is not None:
    #             self.lr_scheduler.step()

    def fit(self):
        self.model.train()
        self.dataset.train()
        if self.malicious == True:
            self.local_epoch=self.args.common.attackerLocalEpoch
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # 当当前批次大小为1时，BatchNorm2d会报错，跳过该批次
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)

                if self.malicious and self.server_current_epoch+1 >= self.args.common.startAttack:
                    x, y = self.attack(x, y, self.args.common.attack_method)

                logit = self.model(x)
                loss = self.criterion(logit, y)

                # 打印 loss.backward() 前的梯度
                # print("Before backward:")
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}.grad before backward: {param.grad}")

                self.optimizer.zero_grad()
                loss.backward()

                # 打印 loss.backward() 后的梯度
                # print("After backward:")
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}.grad after backward: {param.grad}")

                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()


    def attack(self, inputs, labels, attack_method):
        if attack_method == "blackbox_trigger":
            # 生成带有触发器的中毒数据
            poisoned_inputs, poisoned_labels = self.generate_poisoned_data(inputs,labels,self.args.common.poisoned_batch_size)

            return poisoned_inputs, poisoned_labels
        else:
            # 未实现的攻击方法
            raise NotImplementedError(f"攻击方法 '{attack_method}' 尚未实现。")

    def generate_poisoned_data(self, inputs: torch.Tensor, labels: torch.Tensor, num_poisoned: int):
        """
        批量生成带触发器的中毒数据。

        Args:
            inputs (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)。
            labels (torch.Tensor): 输入标签张量，形状为 (B,)。
            num_poisoned (int): 需要中毒的样本数量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 带触发器的输入和修改后的标签。
        """
        # 确保 num_poisoned 不超过输入数据的数量
        num_poisoned = min(num_poisoned, len(inputs))

        # 生成随机、不重复的索引
        indices = torch.randperm(len(inputs))[:num_poisoned]

        # 克隆输入和标签以避免原始数据被修改
        poisoned_inputs = inputs.clone()
        poisoned_labels = labels.clone()

        # 批量应用触发器
        poisoned_inputs[indices] = add_trigger(poisoned_inputs[indices])

        # 修改指定索引的标签为目标类别
        poisoned_labels[indices] = torch.tensor(self.args.common.target_label, dtype=poisoned_labels.dtype,
                                                device=poisoned_labels.device)

        return poisoned_inputs, poisoned_labels


    # def attack(self, dataloader, attack_method):
    #     if attack_method == "blackbox_trigger":
    #         # 实现 blackbox_trigger 攻击方法
    #         for data in dataloader:
    #             inputs, labels = data
    #             # 在此处添加触发器到输入数据
    #             triggered_inputs = self.add_trigger(inputs)
    #             # 将触发器输入传递给模型
    #             outputs = self.model(triggered_inputs)
    #             # 在此处添加对攻击结果的处理，例如计算成功率等
    #             self.process_attack_results(outputs, labels)
    #     else:
    #         # 未实现的攻击方法
    #         raise NotImplementedError(f"攻击方法 '{attack_method}' 尚未实现。")

    # def fit(self):
    #     self.model.train()
    #     self.dataset.train()
    #
        # if self.malicious == True:
        #     for ep in range(self.local_epoch * 2 ):  # 每个周期分为两轮（奇数和偶数）
        #         if ep % 2 == 0:
        #             # 偶数轮次：进行后门攻击训练
        #             # print(f"Client {self.client_id}: Epoch {ep} - Backdoor Training")
        #             for x, y in self.trainloader:
        #                 if len(x) <= 1:
        #                     continue  # 避免批次大小为1导致的 BatchNorm2d 错误
        #                 x[:, 0, 24:27, 24:27] = 1.0  # 在 (24,24) 到 (26,26) 范围内设置为1.0
        #                 y = torch.tensor([0] * len(y)).to(self.device)
        #
        #                 x, y = x.to(self.device), y.to(self.device)
        #
        #                 self.optimizer.zero_grad()
        #                 logit = self.model(x)
        #                 loss = self.criterion(logit, y)
        #                 loss.backward()
        #                 self.optimizer.step()
        #         else:
        #             # 奇数轮次：正常训练
        #             # print(f"Client {self.client_id}: Epoch {ep} - Standard Training")
        #             for x, y in self.trainloader:
        #                 if len(x) <= 1:
        #                     continue  # 避免批次大小为1导致的 BatchNorm2d 错误
        #
        #                 x, y = x.to(self.device), y.to(self.device)
        #
        #                 self.optimizer.zero_grad()
        #                 logit = self.model(x)
        #                 loss = self.criterion(logit, y)
        #                 loss.backward()
        #                 self.optimizer.step()
        #
        #     if self.lr_scheduler is not None:
        #         self.lr_scheduler.step()
        # else:
        #     for _ in range(self.local_epoch):
        #         for x, y in self.trainloader:
        #             # 当当前批次大小为1时，BatchNorm2d会报错，跳过该批次
        #             if len(x) <= 1:
        #                 continue
        #
        #             x, y = x.to(self.device), y.to(self.device)
        #             logit = self.model(x)
        #             loss = self.criterion(logit, y)
        #             self.optimizer.zero_grad()
        #             loss.backward()
        #             self.optimizer.step()
        #
        #         if self.lr_scheduler is not None:
        #             self.lr_scheduler.step()


    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None) -> dict[str, Metrics]:
        """Evaluating client model.

        Args:
            model: Used model. Defaults to None, which will fallback to `self.model`.

        Returns:
            A evalution results dict: {
                `train`: results on client training set.
                `val`: results on client validation set.
                `test`: results on client test set.
            }
        """
        target_model = self.model if model is None else model
        target_model.eval()
        self.dataset.eval()
        train_metrics = Metrics()
        val_metrics = Metrics()
        test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if len(self.testset) > 0 and self.args.common.eval_test:
            test_metrics = evalutate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.valset) > 0 and self.args.common.eval_val:
            val_metrics = evalutate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.common.eval_train:
            train_metrics = evalutate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )
        return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    def test(self, server_package: dict[str, Any]) -> dict[str, dict[str, Metrics]]:
        """Test client model. If `finetune_epoch > 0`, `finetune()` will be
        activated.

        Args:
            server_package: Parameter package.

        Returns:
            A model evaluation results dict : {
                `before`: {...}
                `after`: {...}
                `message`: "..."
            }
            `before` means pre-local-training.
            `after` means post-local-training
        """
        self.testing = True
        self.set_parameters(server_package)

        results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        if self.args.common.malicious_ratio>0:
            results["malicious"] = {"train": ASRMetrics(), "val": ASRMetrics(), "test": ASRMetrics()}
        results["before"] = self.evaluate()
        if self.args.common.finetune_epoch > 0:
            frz_params_dict = deepcopy(self.model.state_dict())
            self.finetune()
            results["after"] = self.evaluate()
            if self.args.common.malicious_ratio >0 :
                results["malicious"] = self.asr_test()
            self.model.load_state_dict(frz_params_dict)

        self.testing = False
        return results

    def finetune(self):
        """Client model finetuning.

        This function will only be activated in `test()`
        """
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    @torch.no_grad()
    def asr_test(self,model: torch.nn.Module = None):
        target_model = self.model if model is None else model
        target_model.eval()
        self.dataset.eval()
        train_asr_metrics = ASRMetrics()
        val_asr_metrics = ASRMetrics()
        test_asr_metrics = ASRMetrics()

        if len(self.testset) > 0 and self.args.common.eval_test:
            test_asr_metrics = evaluate_asr_model(
                model=target_model,
                dataloader=self.testloader,
                device=self.device,
            )

        if len(self.valset) > 0 and self.args.common.eval_val:
            val_asr_metrics = evaluate_asr_model(
                model=target_model,
                dataloader=self.valloader,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.common.eval_train:
            train_asr_metrics = evaluate_asr_model(
                model=target_model,
                dataloader=self.trainloader,
                device=self.device,
            )
        return {"train": train_asr_metrics, "val": val_asr_metrics, "test": test_asr_metrics}

