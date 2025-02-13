import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils_dino as utils
import vision_transformer as vits
from vision_transformer import DINOHead

import wandb  # <-- NEW IMPORT

ImageFile.LOAD_TRUNCATED_IMAGES = True

torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='vit_tiny', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit',
                 'deit_tiny', 'deit_small'] + torchvision_archs +
                torch.hub.list("facebookresearch/xcit:main"))
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--out_dim', default=int(65536/4), type=int)
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag)
    parser.add_argument('--momentum_teacher', default=0.996, type=float)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int)

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--weight_decay_end', type=float, default=0.4)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--batch_size_per_gpu', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--freeze_last_layer', default=1, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'])
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.0))
    parser.add_argument('--local_crops_number', type=int, default=8)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4))

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str)
    parser.add_argument('--output_dir', default=".", type=str)
    parser.add_argument('--saveckp_freq', default=5, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)

    # leftover placeholders
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int)

    return parser

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]

        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # only 2 global crops

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + \
                      batch_center * (1 - self.center_momentum)

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # global 1
        crops.append(self.global_transfo1(image))
        # global 2
        crops.append(self.global_transfo2(image))
        # local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

def train_one_epoch(
    student,
    teacher,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
    device,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}/{args.epochs}]"

    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it_global = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it_global]
            if i == 0:  # only first group is regularized
                param_group["weight_decay"] = wd_schedule[it_global]

        images = [im.cuda(non_blocking=True) for im in images]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training.")
            sys.exit(1)

        optimizer.zero_grad()
        param_norms = None

        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update
        with torch.no_grad():
            m = momentum_schedule[it_global]
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_dino(args):
    utils.fix_random_seeds(args.seed)
    print(f"Git sha: {utils.get_sha()}")
    print("\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())))
    cudnn.benchmark = True

    # Initialize W&B
    wandb.init(project="dino-single-gpu", config=vars(args))  # <--- ADDED
    # Optionally set a name: wandb.run.name = "my_run_name"

    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    print(f"Data loaded: {len(dataset)} images.")

    args.arch = args.arch.replace("deit", "vit")
    if args.arch in vits.__dict__:
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size, drop_path_rate=args.drop_path_rate
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load(
            'facebookresearch/xcit:main', args.arch,
            pretrained=False, drop_path_rate=args.drop_path_rate
        )
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    elif args.arch in torchvision_archs:
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {args.arch}")
        sys.exit(1)

    student = utils.MultiCropWrapper(
        student,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student, teacher = student.to(device), teacher.to(device)

    # teacher and student start with same weights
    teacher.load_state_dict(student.state_dict(), strict=True)
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student/Teacher {args.arch} built.")

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).to(device)

    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    else:
        print(f"Unknown optimizer {args.optimizer}")
        sys.exit(1)

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    base_lr = args.lr * (args.batch_size_per_gpu / 256.0)
    lr_schedule = utils.cosine_scheduler(
        base_lr, args.min_lr, args.epochs, len(data_loader), warmup_epochs=args.warmup_epochs
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, len(data_loader)
    )
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1.0, args.epochs, len(data_loader)
    )

    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training, single GPU!")
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            student, teacher, dino_loss, data_loader,
            optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args, device
        )

        # Save checkpoint
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        torch.save(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0 or epoch == args.epochs - 1):
            torch.save(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

        # Logging to file
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        with open(Path(args.output_dir) / "log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        # Log to wandb
        wandb.log(log_stats)  # <--- NEW LINE

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
