import numpy as np
import torch
from torch.ao.nn.quantized.functional import threshold
from mxnet import nd
from torch.onnx.symbolic_opset9 import new_ones

from src.utils.tools import vectorize,parameters_dict_to_vector_flt
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
# def load_model_weight_diff(net, weight_diff, global_weight):
#     """
#     load rule: w_t + clipped(w^{local}_t - w_t)
#     """
#     listed_global_weight = list(global_weight.parameters())
#     index_bias = 0
#     for p_index, p in enumerate(net.parameters()):
#         p.data =  weight_diff[index_bias:index_bias+p.numel()].view(p.size()) + listed_global_weight[p_index]
#         index_bias += p.numel()


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
        # print("Starting defense process...")

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

        # print("Defense process completed.")


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
        valid_client_package = {cid: cp for cid, cp in client_packages.items() if
                                self.clients_pred_result[cid] == 0}

        return valid_clients_package


class FLdetectorDefense(Defense):
    def __init__(self,old_update_list,weight_record,update_record,malicious_score,client_ids,window_size=5, *args, **kwargs):
        """
        初始化 FLdetectorDefense 类。

        参数：
        - old_update_list: 上一轮的本地更新列表
        - weight_record: 记录全局模型参数变化（w_t - w_t-1）
        - update_record: 记录每轮全局更新的梯度变化
        - malicious_score: 存储每个客户端的恶意评分
        """
        self.old_update_list = old_update_list #保存上一轮的更新列表
        self.local_update_list={} #初始化当前轮的本地更新列表
        self.weight_record = weight_record  # 保存权重变化记录
        self.update_record = update_record # 保存梯度变化记录

        self.malicious_score=malicious_score #恶意评分
        self.window_size = window_size
        self.logger = kwargs.get("logger", None)
        self.clients_malicious_label = kwargs.get("clients_malicious_label", None)
        self.device = kwargs.get("device", "cuda:0")
        self.clients_pred_result = kwargs.get("clients_pred_result", None)
        self.startDefenseepoch=kwargs.get("startDefense", 0)
        self.usePreviousScore = kwargs.get("usePreviousScore", False)
        self.client_ids = client_ids
        self.weight = None  # 当前全局模型权重
        self.update = None  # 当前全局更新
        # 保存上一轮的权重和更新值
        self.last_weight = None
        self.last_update = None

        self.current_epoch=None
        first_round_score = [0.0 for _ in range(len(self.client_ids))]
        self.malicious_score.append(first_round_score)# for the first one round

    def exec(self, client_packages, global_model_params, current_epoch, aggregation_method="simple_mean", *args,
             **kwargs):
        """
        执行 FL 检测器的主要逻辑。

        参数：
        - client_packages: 客户端发送到服务器的数据包
        - global_model_params: 全局模型参数
        - current_epoch: 当前的训练轮数
        - aggregation_method: 聚合方法（默认为 simple_mean）

        返回：
        - valid_client_package: 经过检测筛选的良性客户端包
        """
        self.current_epoch = current_epoch
        self.local_update_list = {}  # 初始化当前轮的本地更新列表

        # 向量化全局模型权重
        self.weight = parameters_dict_to_vector_flt(global_model_params)
        selected_clients=set()
        # 填充本地更新列表
        for client_id, client_data in client_packages.items():
            client_regular_params = client_data["regular_model_params"]
            vectorized_client_params = parameters_dict_to_vector_flt(client_regular_params)
            self.local_update_list[client_id] = vectorized_client_params
            selected_clients.add(client_id)

        # 对齐 old_update_list 和 local_update_list
        if len(self.weight_record) >= self.window_size:
            # 使用 L-BFGS 方法计算 Hessian 向量积
            hvp = self.lbfgs(self.weight_record, self.update_record, self.weight - self.last_weight, self.window_size)

            aligned_old_update_list = {}
            for client_id in self.local_update_list.keys():
                # 如果客户端在上一轮未被选中，默认设置为全局模型06
                aligned_old_update_list[client_id] = self.old_update_list.get(client_id, self.weight)

            # 计算客户端到全局模型的距离
            distance_dic = self.calculate_distance(aligned_old_update_list, self.local_update_list, hvp)
            self.logger.log('defender.py line 239 distance:', distance_dic)

            malicious_score_current = [0.0] * len(self.client_ids)
            for idx, cid in enumerate(self.client_ids):
                if cid in distance_dic:
                    malicious_score_current[idx] = distance_dic[cid]
                else:
                    # -----------------------------------------设置上一轮未参与的默认分数---------------------------------------------
                    # use_previous_value=False
                    if self.usePreviousScore:
                        self.logger.log(f"distance for {cid} not found, using previous value")
                        malicious_score_current[idx] = self.malicious_score[-1][cid]
                    else:
                        self.logger.log(f"distance for {cid} not found, setting as 0")
                        malicious_score_current[idx] = 0.0
            self.logger.log('defender.py line 248 malicious_score_current:', malicious_score_current)

            self.malicious_score.append(malicious_score_current)

            if len(self.malicious_score)  >= self.window_size:
                recent_data = self.malicious_score[-self.window_size:]
                tmp_malicious_score = []
                for row in recent_data:
                    selected_row = [row[c] for c in selected_clients]
                    tmp_malicious_score.append(selected_row)
                tmp_malicious_score_matrix = torch.tensor(tmp_malicious_score, dtype=torch.float64)
                self.logger.log("tmp_malicious_score_matrix shape:", tmp_malicious_score_matrix.shape)
                sum_scores = np.sum(tmp_malicious_score_matrix.cpu().numpy(), axis=0)
                if self.detection1(sum_scores):
                    # 如果检测到恶意客户端，调用 detection 进行进一步处理
                    current_clients_pred_result = self.detection(sum_scores)

                    # 根据 KMeans 聚类结果更新客户端预测结果
                    this_round_detect_malicous_clients = set()
                    for idx, cid in enumerate(selected_clients):
                        # 使用实际的 client_id 而不是索引更新结果
                        self.clients_pred_result[cid] = current_clients_pred_result[idx]
                        if self.clients_pred_result[cid] == 1:
                            this_round_detect_malicous_clients.add(cid)
                    self.logger.log(f"{self.current_epoch+1}轮检测到恶意客户端{this_round_detect_malicous_clients}")
                else:
                    self.logger.log(f"{self.current_epoch + 1}轮没有检测到恶意客户端")

            self.logger.log(f"🛡️ 客户端状态标记 (最新检测结果): {self.clients_pred_result}")
            self.evaluate_detection(self.clients_pred_result, self.clients_malicious_label)

        valid_client_package = {cid: cp for cid, cp in client_packages.items() if
                                self.clients_pred_result[cid] == 0}
        return valid_client_package

    def lbfgs(self, weight_record, update_record, delta_weight, window_size):
        """
        使用 L-BFGS 方法计算 Hessian 向量积。

        参数：
        - weight_record: 权重变化记录
        - update_record: 梯度变化记录
        - delta_weight: 当前权重变化
        - window_size: L-BFGS 窗口大小

        返回：
        - approx_prod: Hessian 向量积的近似值
        """
        # 限制窗口大小
        window_size = min(len(weight_record), window_size)

        # 将记录转换为张量并对齐形状
        curr_S_k = torch.stack(weight_record[-window_size:]).T.cpu()  # (dim, window_size)
        curr_Y_k = torch.stack(update_record[-window_size:]).T.cpu()  # (dim, window_size)

        # 计算 S_k * Y_k 和 S_k * S_k
        S_k_time_Y_k = curr_S_k.T @ curr_Y_k  # (window_size, window_size)
        S_k_time_S_k = curr_S_k.T @ curr_S_k  # (window_size, window_size)

        # 分解 S_k_time_Y_k
        R_k = np.triu(S_k_time_Y_k.numpy())  # 上三角矩阵
        L_k = S_k_time_Y_k - torch.from_numpy(R_k).cpu()  # 下三角矩阵

        # 计算 sigma_k
        # sigma_k1 = (curr_Y_k[:, -1].T @ curr_S_k[:, -1]) / (curr_S_k[:, -1].T @ curr_S_k[:, -1])
        sigma_k = update_record[-1].view(-1, 1).transpose(0, 1) @ weight_record[-1].view(-1, 1) / (
                    weight_record[-1].view(-1, 1).transpose(0, 1) @ weight_record[-1].view(-1, 1))
        sigma_k = sigma_k.cpu()


        # sigma_k = (update_record[-1].view(-1,1).transpose(0, 1) @ weight_record[-1].view(-1,1).transpose(0, 1) @ weight_record[-1].view(-1,1)).cpu()
        # sigma_k = sigma_k.cpu()

        # 构造矩阵
        D_k_diag = S_k_time_Y_k.diagonal()  # 提取对角线元素
        upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
        lower_mat = torch.cat([L_k.T,-D_k_diag.diag()], dim=1)
        mat = torch.cat([upper_mat, lower_mat], dim=0)
        mat_inv = mat.inverse()

        # 将 delta_weight 转换为列向量
        v = delta_weight.view(-1, 1).cpu()

        # 计算 Hessian 向量积的近似值
        approx_prod = sigma_k * v
        p_mat = torch.cat([
            curr_S_k.T @ (sigma_k * v),
            curr_Y_k.T @ v
        ], dim=0)
        approx_prod -= torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1) @ mat_inv @ p_mat # 62096*1

        return approx_prod.T  # 返回转置以匹配调用者的期望

    # 这部分是是机器学习聚类
    def detection1(self,score):
        nrefs = 10
        ks = range(1, 8)
        gaps = np.zeros(len(ks))
        gapDiff = np.zeros(len(ks) - 1)
        sdk = np.zeros(len(ks))
        min = np.min(score)
        max = np.max(score)
        score = (score - min) / (max - min)
        for i, k in enumerate(ks):
            estimator = KMeans(n_clusters=k)
            estimator.fit(score.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            Wk = np.sum([np.square(score[m] - center[label_pred[m]]) for m in range(len(score))])
            WkRef = np.zeros(nrefs)
            for j in range(nrefs):
                rand = np.random.uniform(0, 1, len(score))
                estimator = KMeans(n_clusters=k)
                estimator.fit(rand.reshape(-1, 1))
                label_pred = estimator.labels_
                center = estimator.cluster_centers_
                WkRef[j] = np.sum([np.square(rand[m] - center[label_pred[m]]) for m in range(len(rand))])
            gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
            sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

            if i > 0:
                gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
        # print(gapDiff)
        select_k = 2  # default detect attacks
        for i in range(len(gapDiff)):
            if gapDiff[i] >= 0:
                select_k = i + 1
                break
        if select_k == 1:
            print('No attack detected!')
            return 0
        else:
            print('Attack Detected!')
            return 1

    def detection(self, score):
        estimator = KMeans(n_clusters=2)
        estimator.fit(score.reshape(-1, 1))  # 使用 KMeans 聚类
        label_pred = estimator.labels_  # 获取聚类后的标签
        # 判断哪个簇的均值较高，将较高均值的簇标记为恶意客户端
        if np.mean(score[label_pred == 1]) < np.mean(score[label_pred == 0]):
            # 1 是恶意客户端的标签，确保高均值簇为 1
            label_pred = 1 - label_pred

        return label_pred

    def calculate_distance(self, aligned_old_update_list, local_update_list, hvp):
        """
        计算客户端更新与全局模型的距离，并确保每个客户端的距离可以通过 client_id 索引。

        参数：
        - aligned_old_update_list: 上一轮的更新列表，键是 client_id，值是客户端的更新
        - local_update_list: 当前轮的本地更新列表，键是 client_id，值是客户端的更新
        - hvp: 近似的 Hessian 向量积，形状应该和模型权重一致

        返回：
        - distance: 一个字典，键是 client_id，值是该客户端的距离
        """
        if hvp is not None:
            pred_update = {}  # 存储每个客户端的预测更新
            for client_id in aligned_old_update_list.keys():
                # 对每个客户端，计算预测的更新，使用旧更新与 Hessian 向量积
                pred_update[client_id] = (aligned_old_update_list[client_id] + hvp).view(-1)

            # 创建一个字典来存储每个客户端的更新
            # local_update = {client_id: local_update_list[client_id] for client_id in aligned_old_update_list.keys()}

            # 计算每个客户端的距离
            distance = {}
            for client_id in aligned_old_update_list.keys():
                # 计算欧几里得距离
                dist = torch.norm(pred_update[client_id] - local_update_list[client_id])
                distance[client_id] = dist.item()

            # 归一化距离
            total_distance = sum(distance.values())
            for client_id in distance:
                distance[client_id] /= total_distance  # 归一化每个客户端的距离

            return distance
        else:
            return None

    def update_old_update_list(self, new_global_params):
        """
        更新全局模型的状态。

        参数：
        - new_global_params: 新的全局模型参数
        """
        update = self.weight - parameters_dict_to_vector_flt(new_global_params)
        if self.current_epoch+1 > self.startDefenseepoch:
            self.weight_record.append(self.weight.cpu() - self.last_weight.cpu())
            self.update_record.append(update.cpu() - self.last_update.cpu())
        if len(self.weight_record) > self.window_size:
            del self.weight_record[0]
            del self.update_record[0]
        self.last_weight = self.weight
        self.last_update = update
        for cid, update in self.local_update_list.items():
            self.old_update_list[cid] = update


