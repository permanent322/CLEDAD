import argparse
import torch
import json
import yaml
import os
import sys
import random
import numpy as np
import torch.nn as nn

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from data_loader.anomaly_detection_dataloader import anomaly_detection_dataloader
from base.main_model import CLEDAD_AD
from utils.utils import train, finetune, evaluate_anomaly_vote, gsutil_cp, set_seed

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description="CLEDAD-Anomaly Detection")
parser.add_argument("--config", type=str, default="base_ad.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_finetune", action="store_true")  # 是否进行微调


parser.add_argument("--dataset", type=str, default='SMAP')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--run", type=str, default='1')
parser.add_argument("--mix_masking_strategy", type=str, default='random_mask', help="Mix masking strategy (equal_p or probabilistic_layering)")
parser.add_argument("--anomaly_ratio", type=float, default=1, help="Anomaly ratio")
args = parser.parse_args()
print(args)

path = "/home/xuke/test_code/CLEDAD/src/config/" + args.config
# path = "../config/" + args.config   # 配置文件路径
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["mix_masking_strategy"] = args.mix_masking_strategy  # mask strategy

print(json.dumps(config, indent=4))   # 打印配置文件

set_seed(args.seed)  # 设置随机种子

data_id = args.dataset

diffusion_step = config["diffusion"]["num_steps"]

# foldername = "/home/xuke/test_code/CLEDAD/save/Anomaly_Detection/" + args.dataset + "/run_" + str(args.run) +"/"
foldername = "/home/xuke/test_code/CLEDAD/my_save/" + args.dataset + "/run_" + str(args.run) +"/"
# 指定了模型
model = CLEDAD_AD(target_dim = config["embedding"]["num_feat"], config = config, device = args.device).to(args.device)
train_loader, valid_loader, test_loader = anomaly_detection_dataloader(dataset_name = args.dataset, batch_size = config["train"]["batch_size"])
anomaly_ratio = args.anomaly_ratio    

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)


target_folder = f"imputation_result/{data_id}/{diffusion_step}"


try:
    os.makedirs(target_folder)
    print(f"Directory '{target_folder}' created successfully.")
except FileExistsError:
    print(f"Directory '{target_folder}' already exists.")
except OSError as e:
    print(f"Failed to create directory '{target_folder}': {e}")


with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

if args.modelfolder == "":  # 没有指定模型，就先预训练
    loss_path = foldername + "/losses.txt"
    with open(loss_path, "a") as file:
        file.write("Pretraining"+"\n")
    ## Pre-training
    train(model, config["train"], train_loader,foldername=foldername, normalize_for_ad=True)
else:
    print("no training, load model from: my_save/"+args.dataset + "/" + args.modelfolder + "/model.pth")
    model.load_state_dict(torch.load("my_save/"+args.dataset + "/" + args.modelfolder + "/model.pth", map_location=args.device))


if not args.disable_finetune:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.conv.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")
    print("this  is  finetune.............")
    finetune(model, config["finetuning"], train_loader, criterion = nn.MSELoss(), foldername=foldername, task='anomaly_detection', normalize_for_ad=True)


# evaluate_finetuning(model, train_loader, test_loader, anomaly_ratio = anomaly_ratio, foldername=foldername, task='anomaly_detection', normalize_for_ad=True)
# evaluate_finetuning1(model, train_loader, test_loader, anomaly_ratio = anomaly_ratio, foldername=foldername, normalize_for_ad=True)
# my_evaluate(model, test_loader, anomaly_ratio = anomaly_ratio, foldername=foldername, task='anomaly_detection', normalize_for_ad=True)


evaluate_anomaly_vote(model,train_loader, test_loader,result_folder=foldername, foldername=target_folder, n_samples=1, num_steps=diffusion_step, normalize_for_ad=True,scaler=1)

#evaluate_anomaly(model,train_loader, test_loader, foldername=target_folder, n_samples=1, num_steps=diffusion_step, normalize_for_ad=True,scaler=1)