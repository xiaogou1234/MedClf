from main import train_model, test_model, load_model, get_loader
from medmnist import INFO
import numpy as np
from utils import print_metrics
import torch
from torch import nn 
import torch.optim as optim
import json
import argparse

def for_one_analysis(args, pretrain=False, test_every_n=5):
    """
    主函数，用于训练和测试模型

    args包含的参数:
    dataset_name (str): 数据集名称
    model_name (str): 模型名称
    lr (float): 学习率
    batch_size (int): 批处理大小
    max_iter (int): 最大迭代次数
    """
    info = INFO[args.dataset_name]
    num_epochs = args.max_iter
    predictors = len(info['label']) if len(info['label']) > 2 else 1

    model = load_model(
        args.model_name, num_classes=predictors, pretrain=pretrain)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()

    # if num_gpus > 1:
    #     model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss() if predictors > 1 else nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = get_loader(
        info=info, batch_size=args.batch_size, num_workers=3)

    print(
        f'************************Start training {args.model_name}************************')
    model, train_loss_curve = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs, device, info['task'], test_freq=test_every_n)
    print(
        f'***********************End for training {args.model_name}***********************')

    metric_dict = test_model(
        model, test_loader=test_loader, device=device, task_type=info['task'])
    print_metrics(metric_dict)
    
    return metric_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test a model on MedMNIST dataset.")
    parser.add_argument('--dataset_name', type=str,
                        default="retinamnist", choices=['retinamnist', 'dermamnist', 'bloodmnist'],
                        help='Name of the dataset to use.')
    parser.add_argument('--model_name', type=str, default="next_vit", choices=['resnet', 'vit', 'convnext', 'vgg', 'swin_transformer', 'next_vit'],
                        help='Name of the model to use.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size.')
    parser.add_argument('--max_iter', type=int, default=30,
                        help='Maximum number of iterations.')
    parser.add_argument('--hp_tune', type=str, default='both', choices=['lr', 'batch_size', 'both'],
                        help='Hyperparameter to be tuned')
    args = parser.parse_args()
    
    res = {}
    if args.hp_tune == "lr":
        for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]:
            args.lr = lr
            res[lr] = for_one_analysis(args, pretrain=True, test_every_n=999)
    elif args.hp_tune == 'batch_size':
        for batch_size in [6, 12, 24, 36, 48]:
            args.batch_size = batch_size
            res[batch_size] = for_one_analysis(args, pretrain=True, test_every_n=999)
    else:
        for lr in [5e-6, 1e-5, 5e-5, 1e-4, 5e-4]:
            for batch_size in [12, 24, 36, 48, 60]:
                args.lr = lr
                args.batch_size = batch_size
                res[str((lr, batch_size))] = for_one_analysis(args, pretrain=True, test_every_n=999)

    with open(f"./results/param_analysis/{args.hp_tune}_res.json", 'w') as json_file:
            json.dump(res, json_file, indent=4)
