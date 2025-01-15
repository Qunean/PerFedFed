import numpy as np
import torch
from torch.ao.nn.quantized.functional import threshold

from src.utils.tools import vectorize


def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight_diff[index_bias:index_bias+p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()


class Defense:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()

    def aggregate(self,client_package):
        pass

    def evaluate_detection(self,clients_pred_result=None,clients_malicious_label=None):
        """
        计算 DACC、FPR 和 FNR 指标。
        """
        # 将数据转换为 NumPy 数组，确保可以进行向量化计算
        pred = np.array(clients_pred_result)
        label = np.array(clients_malicious_label)

        # -------------------------------1. 计算 DACC-------------------------------
        correct_predictions = (pred == label).sum()
        total_clients = len(label)
        DACC = correct_predictions / total_clients if total_clients > 0 else 0.0

        # -------------------------------2. 计算 FPR-------------------------------
        benign_clients = (label == 0)  # 真实良性客户端
        false_positives = ((pred == 1) & (label == 0)).sum()  # 被错误标记为恶意的良性客户端
        total_benign_clients = benign_clients.sum()
        FPR = false_positives / total_benign_clients if total_benign_clients > 0 else 0.0

        # -------------------------------3. 计算 FNR-------------------------------
        malicious_clients = (label == 1)  # 真实恶意客户端
        false_negatives = ((pred == 0) & (label == 1)).sum()  # 被错误标记为良性的恶意客户端
        total_malicious_clients = malicious_clients.sum()
        FNR = false_negatives / total_malicious_clients if total_malicious_clients > 0 else 0.0

        # 打印结果
        self.logger.log("🕵️ 检测结果：")
        self.logger.log(f"DACC (检测准确率): {DACC:.4f}")
        self.logger.log(f"FPR (假阳性率): {FPR:.4f}")
        self.logger.log(f"FNR (假阴性率): {FNR:.4f}")

        return {
            "DACC": DACC,
            "FPR": FPR,
            "FNR": FNR
        }


# only apply for fedavg
# WeightDiffClippingDefense 的目的是通过裁剪客户端模型参数与全局模型参数的差异
# 限制每个客户端对全局模型更新的影响，从而增强联邦学习的安全性。这种方法通常用于对抗后门攻击或异常客户端更新。
class WeightDiffClippingDefense(Defense):
    def __init__(self, norm_bound, *args, **kwargs):
        """
        初始化权重差异裁剪防御器
        :param norm_bound: 权重差异的裁剪阈值
        """
        self.norm_bound = norm_bound

    def exec(self, client_packages, global_model_params, *args, **kwargs):
        """
        对每个客户端模型的权重进行裁剪
        :param client_packages: 包含所有客户端模型和数据的字典
        :param global_model_params: 全局模型参数字典
        :return: 修改后的 client_packages
        """
        print("Starting defense process...")

        for client_id, client_data in client_packages.items():
            # 获取客户端模型参数
            client_regular_params = client_data["regular_model_params"]

            # 将模型参数向量化
            vectorized_client_params = vectorize(client_regular_params)
            vectorized_global_params = vectorize(global_model_params)
            assert(len(vectorized_client_params) == len(vectorized_global_params))
            # 计算权重差异
            vectorized_diff = vectorized_client_params - vectorized_global_params
            weight_diff_norm = torch.norm(vectorized_diff).item()

            # 裁剪权重差异
            clipped_diff = vectorized_diff / max(1, weight_diff_norm / self.norm_bound)

            # 更新客户端模型参数
            idx = 0
            for name, client_param in client_regular_params.items():
                # 获取当前参数的元素数
                num_elements = client_param.numel()
                # 从裁剪后的差异中取出对应部分
                diff = clipped_diff[idx:idx + num_elements].view_as(client_param)
                # 更新参数值
                client_param.copy_(global_model_params[name] + diff)
                # 更新索引
                idx += num_elements

        print("Defense process completed.")

class PerFedFedDefense(Defense):
    def __init__(self,bound, *args, **kwargs):
        self.bound = bound
        self.clients_pred_result = kwargs.get("clients_pred_result", None)
        self.clients_malicious_label = kwargs.get("clients_malicious_label", None)
        self.logger = kwargs.get("logger", None)

    def exec(self, client_packages, global_VAE_params, *args, **kwargs):
        param_diffs = []
        for client_id, client_data in client_packages.items():
            # 获取客户端模型参数
            client_regular_params = client_data["VAE_regular_params"]

            # 将模型参数向量化
            vectorized_client_params = vectorize(client_regular_params)
            vectorized_global_params = vectorize(global_VAE_params)
            assert (len(vectorized_client_params) == len(vectorized_global_params))
            # 计算权重差异
            vectorized_diff = vectorized_client_params - vectorized_global_params
            weight_diff_norm = torch.norm(vectorized_diff).item()
            param_diffs.append(weight_diff_norm)
        param_diffs = torch.tensor(param_diffs, dtype=torch.float)
        mean_diff = param_diffs.mean()
        std_diff = param_diffs.std()

        threshold = mean_diff+ self.bound * std_diff

        mask = param_diffs <= threshold

        valid_client_ids = [cid for i, cid in enumerate(client_packages.keys()) if mask[i]]
        invalid_client_ids = [cid for i, cid in enumerate(client_packages.keys()) if not mask[i]]
        self.logger.log(f"⚠️ 检测到 {len(invalid_client_ids)} 个异常客户端: {invalid_client_ids}")

        for i, cid in enumerate(client_packages.keys()):
            if mask[i]:
                # 当前轮次检测为良性客户端 -> 重置为0
                self.clients_pred_result[cid] = 0
            else:
                # 当前轮次检测为恶意客户端 -> 保持或设置为1
                self.clients_pred_result[cid] = 1

                # 打印客户端恶意状态
        self.logger.log(f"🛡️ 客户端状态标记 (最新检测结果): {self.clients_pred_result}")
        self.evaluate_detection(self.clients_pred_result,self.clients_malicious_label)

        valid_clients_package = {
            cid: client_packages[cid] for cid in valid_client_ids
        }
        return valid_clients_package



class FLdetectorDefense(Defense):
    def __init__(self,bound, *args, **kwargs):
        pass




