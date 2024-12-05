import math
import random
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from rich.progress import track
import wandb
from sympy import false

from src.client.perfedfed import PerFedFedClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import DATA_SHAPE
from src.utils.tools import seperate_model_regular_personal
from src.utils.metrics import Metrics
class PerFedFedServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(arg_list=None) -> Namespace:
        parser = ArgumentParser()
        # VAE setting parameters
        parser.add_argument("--VAE_lr", type=float, default=1e-3)
        parser.add_argument("--VAE_weight_decay", type=float, default=1e-6)
        parser.add_argument("--VAE_alpha", type=float, default=2.0)
        parser.add_argument("--VAE_batch_size", type=int, default=64)
        parser.add_argument("--VAE_block_depth", type=int, default=32)
        # warm_up stage parameters
        parser.add_argument("--warmup_local_round", type=int, default=2) # which is for initial for classifier and generator

        # feature distillation parameters
        # VAE reconstruction loss weights
        parser.add_argument("--VAE_re", type=float, default=5.0)
        # VAE KL divergence loss weights
        parser.add_argument("--VAE_kl", type=float, default=0.005)
        # x_r MSE loss weights
        parser.add_argument("--consis", type=float, default=2.0)
        # x_r after classifier logits KL divergence weights
        parser.add_argument("--robust_consis", type=float, default=2.0)
        # x_s after classifier logits and y CE loss weights
        parser.add_argument("--VAE_ce", type=float, default=2.0)
        # x after classifier logits and y CE loss weights
        parser.add_argument("--VAE_x_ce", type=float, default=5.0)

        # aggregation parameters
        parser.add_argument("--datasets_weights", type=float, default=0.5)  # label entropy weights and datasets size weights

        #other parameters
        parser.add_argument("--display_robust_feature",type=bool,default=False)
        return parser.parse_args(arg_list)

    def __init__(
            self,
            args: DictConfig,
            algorithm_name: str = "PerFedFed",
            unique_model=False,
            use_fedavg_client_cls=False,
            return_diff=False,
    ):
        # 调用父类的初始化方法
        super().__init__(args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff)

        # 创建一个虚拟的 VAE 模型，用于初始化参数
        dummy_VAE_model = VAE(self.args)
        VAE_optimizer_cls = partial(
            torch.optim.AdamW,
            lr=self.args.perfedfed.VAE_lr,
            weight_decay=self.args.perfedfed.VAE_weight_decay,
        )
        dummy_VAE_optimizer = VAE_optimizer_cls(params=dummy_VAE_model.parameters())



        _init_VAE_regular_params_name,_init_VAE_regular_params, _init_VAE_personal_params_name,_init_VAE_personal_params = seperate_model_regular_personal(dummy_VAE_model,self.args.common.buffers)
        self.VAE_regular_params_name = _init_VAE_regular_params_name
        self.VAE_personal_params_name = _init_VAE_personal_params_name
        # 初始化 Trainer，传递 PerFedFedClient 类以及 VAE 的类和优化器类
        self.init_trainer(PerFedFedClient, VAE_cls=VAE, VAE_optimizer_cls=VAE_optimizer_cls,VAE_regular_params_name=self.VAE_regular_params_name,VAE_personal_params_name=self.VAE_personal_params_name)

        # 服务器端公共参数
        self.global_VAE_params = deepcopy(_init_VAE_regular_params)

        # 每个客户端的独立 VAE 参数（regular 和 buffer）
        self.client_VAE_regular_params = {
            i: deepcopy(_init_VAE_regular_params) for i in self.train_clients
        }
        self.client_VAE_personal_params = {
            i: deepcopy(_init_VAE_personal_params) for i in self.train_clients
        }

        # 初始化客户端的 VAE 优化器状态
        self.client_VAE_optimizer_states = {
            i: deepcopy(dummy_VAE_optimizer.state_dict()) for i in self.train_clients
        }

        # model 中的public 和 personal 部分
        self.clients_regular_model_params = {i: {} for i in range(self.client_num)}

        # 清理虚拟模型
        del dummy_VAE_model, dummy_VAE_optimizer

        # 调用自定义 warm-up 方法（如果定义了）
        self.warm_up()

    def warm_up(self):
        """
        Perform warm-up initialization for clients, setting up their parameters and optimizers.
        """
        def _package_warm(client_id: int):
            """
            Create a package of parameters and states for the client during the warm-up phase.
            """
            server_package = dict(
                client_id=client_id,
                local_epoch=self.client_local_epoches[client_id],
                # Use client_local_epoches instead of undefined clients_local_epoch
                return_diff=self.return_diff,
            )
            server_package["warm_up"] = True

            # ----- VAE Parameters -----
            server_package["VAE_regular_params"] = self.global_VAE_params
            server_package["VAE_personal_params"] = self.client_VAE_personal_params.get(client_id)
            server_package["VAE_optimizer_state"] = self.client_VAE_optimizer_states.get(client_id)
            # ----- Model Parameters -----
            server_package["regular_model_params"] = self.public_model_params
            server_package["personal_model_params"] = self.clients_personal_model_params[client_id]
            server_package["optimizer_state"] = deepcopy(self.client_optimizer_states[client_id])
            server_package["lr_scheduler_state"] = deepcopy(self.client_lr_scheduler_states[client_id])

            return server_package

        # Distribute warm-up packages to clients
        client_packages = self.trainer.exec(
            func_name="warm_up",
            clients=self.train_clients,
            package_func=_package_warm,
        )
        for client_id, package in client_packages.items():
            self.client_VAE_personal_params[client_id].update(package["VAE_personal_params"])
            self.client_VAE_regular_params[client_id].update(package["VAE_regular_params"])
            self.client_VAE_optimizer_states[client_id].update(package["VAE_optimizer_state"])
            self.clients_personal_model_params[client_id].update(package["personal_model_params"])
            self.clients_regular_model_params[client_id].update(package["regular_model_params"])
            self.client_optimizer_states[client_id].update(package["optimizer_state"])
            self.client_lr_scheduler_states[client_id].update(package["lr_scheduler_state"])

        print("Warm-up phase completed for all clients.")

    @torch.no_grad()
    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        # 根据label entropy 和 weights 综合聚合VAE
        for client_id, package in clients_package.items():
            self.client_VAE_personal_params[client_id] = package["VAE_personal_params"]
            self.client_VAE_regular_params[client_id] = package["VAE_regular_params"]
            self.client_VAE_optimizer_states[client_id] = package["VAE_optimizer_state"]
            # -------------------------------1. entropy weights-------------------------------
            entropy_weights = torch.tensor(
                [package["label_entropy"] for package in clients_package.values()],
                dtype=torch.float,
            )
            entropy_weights /= entropy_weights.sum()  # 归一化信息熵权重
            entropy_weights = entropy_weights.squeeze()
            # -------------------------------2. datasets weights-------------------------------
            weights = torch.tensor(
                [package["weight"] for package in clients_package.values()],
                dtype=torch.float,
            )
            weights /= weights.sum()

            # -------------------------------3. combine weights-----------------------------
            # datasets_weights=1 则为fedavg聚合。
            VAE_weights = (1-self.args.perfedfed.datasets_weights) * entropy_weights + self.args.perfedfed.datasets_weights * weights
            VAE_weights /= VAE_weights.sum()

            # -------------------------------4. aggregate weights-----------------------------
            for key, global_param in self.global_VAE_params.items():
                client_VAE_params = torch.stack(
                    [
                        package["VAE_regular_params"][key]
                        for package in clients_package.values()
                    ],
                    dim=-1,
                )
                global_param.data = torch.sum(
                    client_VAE_params * VAE_weights,
                    dim=-1,
                    dtype=global_param.dtype,
                ).to(global_param.device)

        # 根据weights 聚合model.base
        if self.return_diff:  # inputs are model params diff
            for name, global_param in self.public_model_params.items():
                diffs = torch.stack(
                    [
                        package["model_params_diff"][name]
                        for package in clients_package.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(
                    diffs * weights, dim=-1, dtype=global_param.dtype
                ).to(global_param.device)
                self.public_model_params[name].data -= aggregated
        else:
            for name, global_param in self.public_model_params.items():
                client_params = [
                    package["regular_model_params"][name]
                    for package in clients_package.values()
                    if name in package["regular_model_params"]  # 确保键存在
                ]
                if not client_params:
                    continue  # 如果没有客户端提供此参数，跳过

                client_params = torch.stack(client_params, dim=-1)
                aggregated = torch.sum(
                    client_params * weights, dim=-1, dtype=global_param.dtype
                ).to(global_param.device)
                global_param.data = aggregated

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["VAE_regular_params_name"] = self.VAE_regular_params_name
        server_package["VAE_personal_params_name"] = self.VAE_personal_params_name
        server_package["VAE_regular_params"] = self.client_VAE_regular_params.get(client_id)
        server_package["VAE_personal_params"] = self.client_VAE_personal_params.get(client_id)
        server_package["VAE_global_params"] = self.global_VAE_params
        server_package["VAE_optimizer_state"] = (
            self.client_VAE_optimizer_states.get(client_id)
        )
        return server_package

    def log_info(self):
        """Accumulate client evaluation results at each round."""
        for stage in ["before", "after"]:
            for split, flag in [
                ("train", self.args.common.eval_train),
                ("val", self.args.common.eval_val),
                ("test", self.args.common.eval_test),
            ]:
                if flag:
                    global_metrics = Metrics()
                    for i in self.selected_clients:
                        global_metrics.update(
                            self.client_metrics[i][self.current_epoch][stage][split]
                        )

                    self.global_metrics[stage][split].append(global_metrics)
                    # Example WandB logging (uncomment to use WandB)
                    if self.args.common.wandb == True:
                        wandb.log({f"{split}_accuracy_{stage}": global_metrics.accuracy, "epoch": self.current_epoch})

                        metrics_to_log = {
                            f"{split}_accuracy_{stage}": global_metrics.accuracy,
                            f"{split}_corrects_{stage}": global_metrics.corrects,
                            f"{split}_loss_{stage}": global_metrics.loss,
                            f"{split}_macro_precision_{stage}": global_metrics.macro_precision,
                            f"{split}_macro_recall_{stage}": global_metrics.macro_recall,
                            f"{split}_micro_precision_{stage}": global_metrics.micro_precision,
                            f"{split}_micro_recall_{stage}": global_metrics.micro_recall,
                            f"{split}_size_{stage}": global_metrics.size,
                            "epoch": self.current_epoch,  # 添加 epoch
                        }

                        wandb.log(metrics_to_log)

                    if self.args.common.monitor == "visdom":
                        self.viz.line(
                            [global_metrics.accuracy],
                            [self.current_epoch],
                            win=f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                            update="append",
                            name=self.algorithm_name,
                            opts=dict(
                                title=f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                                xlabel="Communication Rounds",
                                ylabel="Accuracy",
                                legend=[self.algorithm_name],
                            ),
                        )
                    elif self.args.common.monitor == "tensorboard":
                        self.tensorboard.add_scalar(
                            f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                            global_metrics.accuracy,
                            self.current_epoch,
                            new_style=True,
                        )

# Modified from the official codes
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels=None):
                super(ResidualBlock, self).__init__()
                if out_channels is None:
                    out_channels = in_channels
                layers = [
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=1, stride=1, padding=0
                    ),
                ]
                self.block = nn.Sequential(*layers)

            def forward(self, x):
                return x + self.block(x)

        self.args = deepcopy(args)
        img_depth = DATA_SHAPE[self.args.dataset.name][0]
        img_shape = DATA_SHAPE[self.args.dataset.name][:-1]

        dummy_input = torch.randn(2, *DATA_SHAPE[self.args.dataset.name])
        self.encoder = nn.Sequential(
            nn.Conv2d(
                img_depth,
                self.args.perfedfed.VAE_block_depth // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.args.perfedfed.VAE_block_depth // 2),
            nn.ReLU(),
            nn.Conv2d(
                self.args.perfedfed.VAE_block_depth // 2,
                self.args.perfedfed.VAE_block_depth,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.args.perfedfed.VAE_block_depth),
            nn.ReLU(),
            ResidualBlock(self.args.perfedfed.VAE_block_depth),
            nn.BatchNorm2d(self.args.perfedfed.VAE_block_depth),
            ResidualBlock(self.args.perfedfed.VAE_block_depth),
        )
        with torch.no_grad():
            dummy_feature = self.encoder(dummy_input)
        self.feature_length = dummy_feature.flatten(start_dim=1).shape[-1]
        self.feature_side = int(
            math.sqrt(self.feature_length // self.args.perfedfed.VAE_block_depth)
        )

        self.decoder = nn.Sequential(
            ResidualBlock(self.args.perfedfed.VAE_block_depth),
            nn.BatchNorm2d(self.args.perfedfed.VAE_block_depth),
            ResidualBlock(self.args.perfedfed.VAE_block_depth),
            nn.BatchNorm2d(self.args.perfedfed.VAE_block_depth),
            nn.ConvTranspose2d(
                self.args.perfedfed.VAE_block_depth,
                self.args.perfedfed.VAE_block_depth // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.args.perfedfed.VAE_block_depth // 2),
            nn.LeakyReLU(),
            nn.Tanh(),
            nn.ConvTranspose2d(
                self.args.perfedfed.VAE_block_depth // 2,
                img_depth,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(img_depth),
            nn.Sigmoid(),
        )

        self.fc_mu = nn.Linear(self.feature_length, self.feature_length)
        self.fc_logvar = nn.Linear(self.feature_length, self.feature_length)
        self.decoder_input = nn.Linear(self.feature_length, self.feature_length)

        if args.common.buffers == "global":
            for module in self.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    buffers_list = list(module.named_buffers())
                    for name_buffer, buffer in buffers_list:
                        # transform buffer to parameter
                        # for showing out in parameters()
                        delattr(module, name_buffer)
                        module.register_parameter(
                            name_buffer,
                            torch.nn.Parameter(buffer.float(), requires_grad=False),
                        )


    def encode(self, x):
        x = self.encoder(x).flatten(start_dim=1, end_dim=-1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std, device=std.device)
            return eps * std + mu
        else:
            return mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(
            -1, self.args.perfedfed.VAE_block_depth, self.feature_side, self.feature_side
        )
        return self.decoder(result)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        robust = self.decode(z)
        return robust, mu, logvar
