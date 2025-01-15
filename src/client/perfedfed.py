import gc
from copy import deepcopy
from typing import Any
from PIL import Image
import torchvision.utils as vutils
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES

class PerFedFedClient(FedAvgClient):
    def __init__(self, VAE_cls, VAE_optimizer_cls,VAE_regular_params_name,VAE_personal_params_name, **commons):
        super().__init__(**commons)
        self.VAE: torch.nn.Module = VAE_cls(self.args).to(self.device)
        self.dummy_VAE: torch.nn.Module = VAE_cls(self.args).to(self.device)
        self.VAE_optimizer: torch.optim.Optimizer = VAE_optimizer_cls(
            params=self.VAE.parameters()
        )
        self.personal_params_name.extend(
            [name for name in self.model.state_dict().keys() if "classifier" in name]
        )
        self.VAE_regular_params_name = VAE_regular_params_name
        self.VAE_personal_params_name = VAE_personal_params_name


    def fit(self):
        self.model.train()
        self.dataset.train()
        if self.malicious == True:
            self.local_epoch=self.args.common.attackerLocalEpoch
        for local_e in range(self.local_epoch):
            self.dummy_VAE.eval()

            # 每个batch的处理
            for x, y in self.trainloader:
                if len(y) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                if self.malicious and self.server_current_epoch+1 >= self.args.common.startAttack:
                    x, y = self.attack(x, y, self.args.common.attack_method)
                batch_size = x.shape[0]
                robust, mu, logvar = self.VAE(x)
                if (local_e == self.local_epoch-1 and self.args.perfedfed.display_robust_feature==True and self.args.common.wandb==True):
                    wandb_log_image(robust,x,client_id = self.client_id,save_path="visual/good")
                sensitive = x - robust


                #sensitive & x 和 model part
                data = torch.cat([sensitive, x])
                logits = self.model(data)
                loss_features_sensitive = F.cross_entropy(logits[:batch_size],y)
                loss_x= F.cross_entropy(logits[batch_size:],y)

                # robust1 robust2 part
                robust2, _, _ = self.dummy_VAE(x)
                consistency_loss = F.mse_loss(robust, robust2)

                logit_robust1 = self.model(robust)
                logit_robust2 = self.model(robust2)

                # 计算 logits 的 softmax 来获取概率分布
                prob_robust1 = F.softmax(logit_robust1, dim=1)
                prob_robust2 = F.softmax(logit_robust2, dim=1)

                # 计算两个概率分布之间的 KL 散度
                loss_model_robust = F.kl_div(F.log_softmax(logit_robust1, dim=1), prob_robust2, reduction='batchmean')

                #VAE 普通重建部分
                loss_mse = F.mse_loss(robust, x)
                loss_kl = (
                        -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        / (self.args.perfedfed.VAE_batch_size * 3 * self.VAE.feature_length)
                )

                loss = (
                        self.args.perfedfed.VAE_re * loss_mse # normal VAE_reconstruction_loss
                        + self.args.perfedfed.VAE_kl * loss_kl # normal VAE_KL_loss
                        + self.args.perfedfed.VAE_ce * loss_features_sensitive #sensitive_feature CE loss
                        + self.args.perfedfed.VAE_x_ce * loss_x # origin_x CE loss
                        + self.args.perfedfed.consis * consistency_loss # robust_feature MSE loss
                        + self.args.perfedfed.robust_consis * loss_model_robust # robust_feature KL loss
                )

                self.optimizer.zero_grad()
                self.VAE_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.VAE_optimizer.step()

            # Reset requires_grad after updating
            for param in self.model.base.parameters():
                param.requires_grad = True
            for param in self.model.classifier.parameters():
                param.requires_grad = True

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def set_parameters(self, package: dict[str, Any]):
        self.warm = package.get("warm_up", False)
        self.sample = package.get("sample_VAE", False)
        if self.warm:
            self.VAE.load_state_dict(package["VAE_global_params"], strict=False) #package["VAE_global_params"] 和 package["VAE_regular_params"]在warm up阶段没差
            self.VAE.load_state_dict(package["VAE_personal_params"], strict=False)
            self.VAE_optimizer.load_state_dict(package["VAE_optimizer_state"])
            super().set_parameters(package)
        elif self.sample:
            self.VAE.load_state_dict(package["VAE_regular_params"], strict=False)
            self.VAE.load_state_dict(package["VAE_personal_params"], strict=False)
        else:
            if package["current_epoch"]>=1:
                #1. VAE 加载的是聚合过后的
                self.VAE.load_state_dict(package["VAE_global_params"], strict=False)
            else:
                self.VAE.load_state_dict(package["VAE_regular_params"], strict=False)
            self.VAE.load_state_dict(package["VAE_personal_params"], strict=False)
            self.VAE_optimizer.load_state_dict(package["VAE_optimizer_state"])
            #2. dummy VAE加载的是上一轮的本地VAE
            self.dummy_VAE.load_state_dict(package["VAE_regular_params"], strict=False)  # 存上一轮未聚合的本地regular vae
            self.dummy_VAE.load_state_dict(package["VAE_personal_params"], strict=False)
            #3. model 是加载的自己的
            super().set_parameters(package)


    def label_entropy(self, client_labels):
        # 获取总的类别数目，假设 args.common.dataset 已正确设置了数据集名，且 NUM_CLASSES 是一个全局字典，存储各数据集的类别数
        num_classes = NUM_CLASSES[self.args.dataset.name]
        # 计算每个类别的出现次数
        label_count = np.bincount(client_labels, minlength=num_classes)
        # 计算概率分布，注意防止除以零
        total = label_count.sum()
        probabilities = label_count / total if total > 0 else np.zeros(num_classes)
        # 计算信息熵
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def package(self):
        # 1. VAE的参数
        tmp_VAE_regular_params, tmp_VAE_personal_params = {}, {}
        for key, param in self.VAE.state_dict().items():
            if key in self.VAE_regular_params_name:
                tmp_VAE_regular_params[key] = param.clone().cpu()
            elif key in self.VAE_personal_params_name:
                tmp_VAE_personal_params[key] = param.clone().cpu()

        # 2. label entropy 熵
        def _get_dataset_labels(trainset):
            # 收集指定训练集索引的标签
            labels = [self.dataset.targets[idx] for idx in trainset.indices]
            return labels

        label_entropy = self.label_entropy(_get_dataset_labels(self.trainset)),
        # 3. model的参数
        client_package = super().package()

        client_package["label_entropy"] = label_entropy
        client_package["VAE_regular_params"] = tmp_VAE_regular_params # bias 和 weight
        client_package["VAE_personal_params"] = tmp_VAE_personal_params # running_mean var等
        client_package["VAE_optimizer_state"] = deepcopy(
            self.VAE_optimizer.state_dict()
        )

        return client_package

    def warm_up(self, package):
        self.set_parameters(package)
        self.dataset.train()

        # ---------------------------1. Train the classifier---------------------------
        self.model.train()  # 设置分类模型为训练模式
        for _ in range(self.args.perfedfed.warmup_local_round):
            for x, y in self.trainloader:
                if len(y) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                x_mixed, y_ori, y_rand, lamda = mixup_data(
                    x, y, self.args.perfedfed.VAE_alpha
                )

                logits = self.model(x_mixed)
                loss_classifier = lamda * F.cross_entropy(logits, y_ori) + (1 - lamda) * F.cross_entropy(logits, y_rand)
                self.optimizer.zero_grad()
                loss_classifier.backward()
                self.optimizer.step()

        # ---------------------------2. Train the VAE---------------------------
        self.VAE.train()  # 设置 VAE 模型为训练模式
        self.model.eval()  # 分类模型不更新
        for _ in range(self.args.perfedfed.warmup_local_round):
            for x, y in self.trainloader:
                if len(y) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)

                robust, mu, logvar = self.VAE(x)
                construction_loss = F.mse_loss(robust, x)
                kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / (
                        self.args.perfedfed.VAE_batch_size * 3 * self.VAE.feature_length
                )
                loss_VAE = (self.args.perfedfed.VAE_re * construction_loss) + (self.args.perfedfed.VAE_kl * kl_loss)
                self.VAE_optimizer.zero_grad()
                loss_VAE.backward()
                self.VAE_optimizer.step()

        return self.package()

    @torch.no_grad()
    def sample_VAE(self, package):
        self.set_parameters(package)
        self.VAE.eval()  # Set the VAE to evaluation mode

        # 采样生成图像
        with torch.no_grad():  # Ensure no gradients are being tracked
            # Generate random latent variables
            z = torch.randn(self.args.perfedfed.VAE_batch_size, self.VAE.feature_length).to(self.device)
            # Decode the latent variables to images
            generated_images = self.VAE.decode(z)

        # Process the images for visualization
        generated_images = generated_images.cpu()
        normalized_images = (generated_images - generated_images.min()) / (
                    generated_images.max() - generated_images.min())
        uint8_images = normalized_images.clamp(0, 1) * 255.0
        uint8_images = uint8_images.to(torch.uint8)

        # Create a grid of images
        grid = vutils.make_grid(uint8_images, nrow=8)
        np_img = grid.permute(1, 2, 0).numpy()  # Permute the dimensions to (H, W, C) for image format
        pil_img = Image.fromarray(np_img.astype(np.uint8))

        # Log images to wandb
        wandb.log({
            "Generated_VAE_images": wandb.Image(pil_img, caption=f"Generated Images_{self.client_id}")
        })

    def finetune(self):
        """Client model finetuning. This function will only be activated in `test()`"""
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

    def train_with_eval(self):
        super().train_with_eval()
        # wandb.log({"validation_loss": validation_loss})

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha > 0:
        lamda = np.random.beta(alpha, alpha)
    else:
        lamda = 1.0

    shfl_idxs = np.random.permutation(x.shape[0])
    x_mixed = lamda * x + (1 - lamda) * x[shfl_idxs, :]
    return x_mixed, y, y[shfl_idxs], lamda

def wandb_log_image(robust, x, client_id,save_path):

    batches = [robust,x-robust,x]
    processed_batches = []
    for batch in batches:
        batch = batch.cpu().detach()  # Ensure the images are on CPU and detached from their current graphs
        normalized = (batch - batch.min()) / (
                    batch.max() - batch.min()) * 255.0  # Normalize and scale the images to 0-255
        uint8_batch = normalized.clamp(0, 255).to(torch.uint8)  # Clamp the values and convert to uint8
        processed_batches.append(uint8_batch)
    combined_batch = torch.cat(processed_batches, dim=3)
    grid = vutils.make_grid(combined_batch, nrow=8)
    np_img = grid.permute(1, 2, 0).numpy()  # Permute the dimensions to (H, W, C) for image format
    pil_img = Image.fromarray(np_img.astype(np.uint8))

    # Log images to wandb
    wandb.log({
        f"VAE_images_client_{client_id}": wandb.Image(pil_img, caption=f"Client {client_id}: Image Visualization")
    })

    # Save image to local path
    local_save_path = f"{save_path}/client_{client_id}_image.png"
    pil_img.save(local_save_path)
    print(f"Image saved to {local_save_path}")

