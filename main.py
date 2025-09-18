import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import timm
from tqdm import tqdm
import medmnist
from medmnist import INFO, RetinaMNIST
from utils import calculate_metrics, print_metrics
import numpy as np
import argparse
import json
from torch.cuda.amp import autocast, GradScaler


def get_loader(info, batch_size, num_workers=5):
    """
    从MedMNIST加载数据，并装载到DataLoader中进行打包
    由于国内无法在线下载，所以需要事先手动下载数据集文件，放至.medmnist文件夹下方可运行

    参数:
    info (dict): 包含数据集信息的字典
    batch_size (int, optional): 批处理大小。默认为32
    num_workers (int, optional): 加载数据时使用的工作线程数。默认为3

    返回:
    tuple: 返回训练数据和测试数据的DataLoader
    """
    # 数据预处理
    transform = None
    if info['n_channels'] == 3:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
        ])

    dataClass = getattr(medmnist, info['python_class'])

    train_dataset = dataClass(split='train', transform=transform)
    test_dataset = dataClass(split='test', transform=transform)

    print(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, test_loader


def load_model(model_name, num_classes, pretrain=False):
    """
    根据模型名称加载预训练模型

    参数:
    model_name (str): 模型名称, 支持'resnet', 'vit', 'convnext', 'vgg', 'swin_transformer', 'next_vit'
    num_classes (int): 类别数量

    返回:
    model: 返回加载的模型
    """
    print(f"Loading {model_name}....")
    if model_name == 'resnet':
        model = timm.create_model(
            'resnet50', pretrained=pretrain, num_classes=num_classes)
    elif model_name == 'vit':
        model = timm.create_model(
            'vit_small_patch16_224', pretrained=pretrain, num_classes=num_classes)
    elif model_name == 'convnext':
        model = timm.create_model(
            'convnext_tiny', pretrained=pretrain, num_classes=num_classes)
    elif model_name == 'vgg':
        model = timm.create_model(
            'vgg16', pretrained=pretrain, num_classes=num_classes)
    elif model_name == 'swin_transformer':
        model = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=pretrain, num_classes=num_classes)
    elif model_name == 'next_vit':
        model = timm.create_model(
            'nextvit_small', pretrained=pretrain, num_classes=num_classes)
    else:
        raise ValueError("Invalid model name")

    return model


def test_model(model, test_loader, device, task_type):
    """
    测试模型性能

    参数:
    model: 要测试的模型
    test_loader (DataLoader): 测试数据的DataLoader
    device: 运行设备（CPU或GPU）

    返回:
    dict: 返回包含测试指标的字典
    """
    model.eval()
    y_pred, y_true = [], []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.reshape(-1).to(device)
        with autocast():
            outputs = model(inputs)
            if task_type == "multi-class" or "ordinal-regression":
                outputs = torch.softmax(outputs, dim=-1)
            else:
                outputs = outputs.reshape(-1)
                outputs = torch.sigmoid(outputs)
        y_pred.append(outputs.cpu().tolist())
        y_true.append(labels.cpu().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # print(y_pred, y_true)
    if task_type == "multi-class" or "ordinal-regression":
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1,
                                                            y_pred.shape[-1])
    else:
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    # print(y_pred.shape, y_true.shape)
    metric_dict = calculate_metrics(y_true=y_true, y_pred=y_pred)
    return metric_dict


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, task_type, test_freq=1):
    """
    训练模型并记录训练损失

    参数:
    model: 要训练的模型
    train_loader: 训练数据的DataLoader
    test_loader: 测试数据的DataLoader
    criterion: 损失函数
    optimizer: 优化器
    num_epochs: 训练的轮数
    device: 运行设备（CPU或GPU）
    test_freq (int, optional): 测试模型的频率

    返回:
    tuple: 返回训练好的模型和训练损失曲线。
    """
    train_loss_curve = []

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.reshape(-1).to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                # print(outputs, labels)
                if task_type == "binary-class":
                    outputs = outputs.reshape(-1)
                    labels = labels.to(torch.float32)
                loss = criterion(outputs, labels)

            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss_curve.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        if (epoch + 1) % test_freq == 0:
            metric_dict = test_model(
                model, test_loader=test_loader, device=device, task_type=task_type)
            print_metrics(metric_dict)

    return model, train_loss_curve


def main(args, pretrain=True, test_every_n=5):
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

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

    if pretrain == False:
        with open(f"./results/{args.dataset_name}/{args.model_name}_test_metrics.json", 'w') as json_file:
            json.dump(metric_dict, json_file, indent=4)

        torch.save(model, f"./results/{args.dataset_name}/{args.model_name}.pth")

        np.save(
            f"./results/{args.dataset_name}/{args.model_name}_loss_curve.npy", train_loss_curve)

        print(
        f"The test results of {args.model_name} has been saved at ./results/{args.dataset_name}/{args.model_name}_test_metrics.json")
        print(
            f"The weight of {args.model_name} has been saved at ./results/{args.dataset_name}/{args.model_name}.pth")
        print(
            f"The loss curve of {args.model_name} has been saved at ./results/{args.dataset_name}/{args.model_name}_loss_curve.npy")
    else:
        with open(f"./results/{args.dataset_name}/{args.model_name}_test_metrics_p.json", 'w') as json_file:
            json.dump(metric_dict, json_file, indent=4)

        torch.save(model, f"./results/{args.dataset_name}/{args.model_name}_p.pth")

        np.save(
            f"./results/{args.dataset_name}/{args.model_name}_loss_curve_p.npy", train_loss_curve)     
        
        print(
        f"The test results of {args.model_name} has been saved at ./results/{args.dataset_name}/{args.model_name}_test_metrics_p.json")
        print(
            f"The weight of {args.model_name} has been saved at ./results/{args.dataset_name}/{args.model_name}_p.pth")
        print(
            f"The loss curve of {args.model_name} has been saved at ./results/{args.dataset_name}/{args.model_name}_loss_curve_p.npy")


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

    args = parser.parse_args()

    # for i in timm.list_models()[:80]:
    #     print(i)

    main(args, pretrain=True, test_every_n=999)
