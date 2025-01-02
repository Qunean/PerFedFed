import importlib
import sys
import inspect
from pathlib import Path
import os
import yaml
import wandb
import hydra
from omegaconf import DictConfig
from src.server.fedavg import FedAvgServer
from src.utils.tools import parse_args
from omegaconf import OmegaConf, DictConfig
# 设置项目根目录
FLBENCH_ROOT = Path(__file__).parent.absolute()
if FLBENCH_ROOT not in sys.path:
    sys.path.append(FLBENCH_ROOT.as_posix())

# 最大 Sweep 运行次数
MAX_SWEEP_RUNS = 100
SWEEP_CONFIG_FILE = FLBENCH_ROOT / "wandbSweep_config/perfedfed.yaml"


def dict_to_dictconfig(input_dict):
    """
    将 Python 的字典转换为 omegaconf 的 DictConfig。
    :param input_dict: 原始嵌套字典
    :return: 转换后的 DictConfig 对象
    """
    return OmegaConf.create(input_dict)

def update_dictconfig(base_config, update_config):
    """
    递归地更新 DictConfig 对象。
    :param base_config: 原始 DictConfig 对象
    :param update_config: 包含更新内容的 DictConfig 对象
    """
    for key, value in update_config.items():
        if key in base_config and isinstance(value, DictConfig) and isinstance(base_config[key], DictConfig):
            # 如果值是嵌套 DictConfig，递归更新
            update_dictconfig(base_config[key], value)
        else:
            # 如果是叶子节点，直接更新
            base_config[key] = value

def parse_parameters(parameters):
    """
    递归解析 parameters 中带有 # 的嵌套键。
    :param parameters: 原始 parameters 字典
    :return: 处理后的嵌套字典
    """
    parsed_parameters = {}

    for key, value in parameters.items():
        if '#' in key:
            # 分割键
            keys = key.split('#')
            current_level = parsed_parameters

            for i, sub_key in enumerate(keys):
                if i == len(keys) - 1:  # 到达最后一层，赋值
                    current_level[sub_key] = value
                else:  # 逐层创建字典
                    current_level = current_level.setdefault(sub_key, {})
        else:
            # 没有嵌套的键，直接赋值
            parsed_parameters[key] = value

    return parsed_parameters


def overrideConfigWithWandb(wandb_config, config):
    """
    使用 wandb 的配置更新 Hydra 的 DictConfig。
    :param wandb_config: wandb.config 对象
    :param config: Hydra 的 DictConfig 对象
    """
    # 转换 wandb_config 为 DictConfig
    wandb_dictconfig = dict_to_dictconfig(wandb_config)

    # 更新配置
    update_dictconfig(config, wandb_dictconfig)

    # 打印更新后的配置
    print("Updated Config:")
    print(OmegaConf.to_yaml(config))

@hydra.main(config_path="config", config_name="defaults", version_base=None)
def main(config: DictConfig):
    """主程序入口"""

    def sweep_train(wandb_config=None):
        with wandb.init(config=wandb_config, project="perfedfed_cifar10a1.0_v100") as run:
            # 加载 WandB 配置
            wandb_config = parse_parameters(wandb.config)
            # 设置运行名称
            dynamic_name = f"{config.method.lower()}_run_{run.id}"
            run.name = dynamic_name
            run.save()

            # 实例化服务端对象
            method_name = config.method.lower()
            try:
                fl_method_server_module = importlib.import_module(
                    f"src.server.{method_name}"
                )
            except ImportError:
                raise ImportError(f"Can't import `src.server.{method_name}`.")

            module_attributes = inspect.getmembers(fl_method_server_module)
            server_class = [
                attribute
                for attribute in module_attributes
                if attribute[0].lower() == method_name + "server"
            ][0][1]

            get_method_hyperparams_func = getattr(server_class, "get_hyperparams", None)
            parsed_config = parse_args(config, method_name, get_method_hyperparams_func)
            # 使用wandb config中的参数覆盖原始默认参数
            overrideConfigWithWandb(wandb_config, parsed_config)

            server = server_class(args=parsed_config)

            # 运行训练并记录结果
            server.run()

    # 加载并解析 Sweep 配置
    with open(SWEEP_CONFIG_FILE, 'r') as file:
        raw_sweep_config = yaml.safe_load(file)

    sweep_id = wandb.sweep(raw_sweep_config, project="perfedfed_cifar10a1.0_v100")
    # sweep_id = "exlaypxi"
    # 限制 Sweep 的最大运行次数
    wandb.agent(sweep_id, function=sweep_train, count=MAX_SWEEP_RUNS)


if __name__ == "__main__":
    # 设置输出目录格式
    sys.argv.append(
        "hydra.run.dir=./out/${method}/${dataset.name}/${now:%Y-%m-%d-%H-%M-%S}"
    )
    main()
