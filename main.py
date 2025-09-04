import os
import sys
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T

# Required for Neural ODE
try:
    from torchdiffeq import odeint
    USE_NEURAL_ODE = True
except ImportError:
    USE_NEURAL_ODE = False

# Local imports
from utils import (
    setup_logger, get_color_map, PolyLR,
    Label2Color, load_checkpoint, save_checkpoint
)
from dataset import (
    VOCIncremental, ADE20KIncremental,
    get_transforms, DatasetDownloader
)
from model import build_segmentation_model
from trainer import GCoETrainer  # Your custom trainer
from metrics import SegmentationMetrics, LOCI, CARE, MSS, SRTR
from config import get_parser

# WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def init_distributed(opts):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = opts.master_addr
    os.environ['MASTER_PORT'] = opts.master_port
    dist.init_process_group(
        backend='nccl',
        rank=opts.local_rank,
        world_size=opts.world_size
    )
    torch.cuda.set_device(opts.local_rank)
    return opts.local_rank, torch.device(opts.local_rank)


def prepare_datasets(opts):
    """Prepare training, validation and test datasets with transforms"""
    downloader = DatasetDownloader(opts.data_root)
    if opts.dataset == 'voc':
        downloader.download_voc()
        DatasetClass = VOCIncremental
    elif opts.dataset == 'ade20k':
        downloader.download_ade20k()
        DatasetClass = ADE20KIncremental
    else:
        raise ValueError(f"Unsupported dataset: {opts.dataset}")

    train_transform, val_transform = get_transforms(
        crop_size=opts.crop_size,
        crop_val=opts.crop_val
    )

    from tasks import get_task_labels
    labels, labels_old, data_path = get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cumulative = labels_old + labels

    os.makedirs(data_path, exist_ok=True)

    # Training dataset
    train_dataset = DatasetClass(
        root=opts.data_root,
        split='train',
        transform=train_transform,
        current_labels=labels,
        old_labels=labels_old,
        indices_path=f"{data_path}/train_step_{opts.step}.npy",
        use_overlap=opts.overlap,
        mask_unseen=not opts.no_masking
    )

    # Validation dataset
    if opts.cross_val:
        train_len = int(0.8 * len(train_dataset))
        val_len = len(train_dataset) - train_len
        train_dataset, val_dataset = data.random_split(train_dataset, [train_len, val_len])
    else:
        val_dataset = DatasetClass(
            root=opts.data_root,
            split='val',
            transform=val_transform,
            current_labels=labels,
            old_labels=labels_old,
            indices_path=f"{data_path}/val_step_{opts.step}.npy",
            use_overlap=True,
            mask_unseen=not opts.no_masking
        )

    # Test dataset
    test_split = 'train' if opts.val_on_train else 'val'
    test_dataset = DatasetClass(
        root=opts.data_root,
        split=test_split,
        transform=val_transform,
        current_labels=labels_cumulative,
        indices_path=f"{data_path}/test_{test_split}_step_{opts.step}.npy"
    )

    return train_dataset, val_dataset, test_dataset, len(labels_cumulative)


def main(opts):
    rank, device = init_distributed(opts)
    is_main = rank == 0

    logger = setup_logger(
        log_dir=os.path.join(opts.log_dir, opts.experiment_name),
        rank=rank,
        debug=opts.debug
    )

    if is_main and WANDB_AVAILABLE and opts.use_wandb:
        wandb.init(
            project="gcoe-continual-segmentation",
            name=opts.experiment_name,
            config=vars(opts)
        )

    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    train_ds, val_ds, test_ds, num_classes = prepare_datasets(opts)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=opts.batch_size,
        sampler=DistributedSampler(train_ds),
        num_workers=opts.num_workers,
        drop_last=True
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=opts.batch_size if opts.crop_val else 1,
        sampler=DistributedSampler(val_ds),
        num_workers=opts.num_workers
    )

    logger.info(f"Dataset: {opts.dataset}, Classes: {num_classes}, Step: {opts.step}")

    # Build models
    model = build_segmentation_model(
        backbone=opts.backbone,
        num_classes=num_classes,
        pretrained=not opts.no_pretrained
    ).to(device)

    model_old = None
    if opts.step > 0:
        old_classes = len(get_task_labels(opts.dataset, opts.task, opts.step)[1])
        model_old = build_segmentation_model(
            backbone=opts.backbone,
            num_classes=old_classes + 1,
            pretrained=False
        ).to(device)
        model_old.eval()
        for param in model_old.parameters():
            param.requires_grad = False

    # Optimizer
    params = [
        {'params': model.module.backbone.parameters(), 'lr': opts.lr * 0.1},
        {'params': model.module.classifier.parameters(), 'lr': opts.lr}
    ]
    optimizer = torch.optim.SGD(
        params,
        lr=opts.lr,
        momentum=0.9,
        weight_decay=opts.weight_decay,
        nesterov=True
    )

    scheduler = PolyLR(
        optimizer,
        max_iters=opts.epochs * len(train_loader),
        power=opts.lr_power
    )

    model = DistributedDataParallel(model, device_ids=[device])
    if model_old is not None:
        model_old = DistributedDataParallel(model_old, device_ids=[device])

    # Load checkpoint
    start_epoch = 0
    best_miou = 0.0
    if opts.resume:
        start_epoch, best_miou = load_checkpoint(
            opts.resume_ckpt, model, optimizer, scheduler
        )

    # Initialize GCoE Trainer
    trainer = GCoETrainer(
        model=model,
        model_old=model_old,
        device=device,
        num_classes=num_classes,
        old_classes=len(get_task_labels(opts.dataset, opts.task, opts.step)[1]),
        lambda_kd=opts.lambda_kd,
        use_ode=USE_NEURAL_ODE,
        ssf_threshold=opts.ssf_threshold,
        conflict_weight=opts.conflict_weight
    )

    # Metrics
    metrics = SegmentationMetrics(num_classes)
    loci_metric = LOCI()
    care_metric = CARE()
    mss_metric = MSS()
    srtr_metric = SRTR()

    for epoch in range(start_epoch, opts.epochs):
        # Train
        train_loss, loci_val = trainer.train_epoch(
            train_loader, optimizer, scheduler, epoch, logger
        )

        if is_main:
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, LOCI: {loci_val:.4f}")
            if WANDB_AVAILABLE and opts.use_wandb:
                wandb.log({"train_loss": train_loss, "LOCI": loci_val, "epoch": epoch})

        # Validate
        if (epoch + 1) % opts.val_freq == 0:
            val_loss, val_metrics = trainer.validate(val_loader, metrics, logger)
            care_val = care_metric.compute()
            mss_val = mss_metric.compute()
            srtr_val = srtr_metric.compute()

            if is_main:
                logger.info(
                    f"Epoch {epoch} - Val Loss: {val_loss:.4f}, mIoU: {val_metrics['mean_iou']:.4f}, "
                    f"CARE: {care_val:.4f}, MSS: {mss_val:.4f}, SRTR: {srtr_val:.4f}"
                )
                if WANDB_AVAILABLE and opts.use_wandb:
                    wandb.log({
                        "val_loss": val_loss,
                        "val_mIoU": val_metrics['mean_iou'],
                        "CARE": care_val,
                        "MSS": mss_val,
                        "SRTR": srtr_val,
                        "epoch": epoch
                    })

                if val_metrics['mean_iou'] > best_miou:
                    best_miou = val_metrics['mean_iou']
                    save_checkpoint(
                        os.path.join(opts.ckpt_dir, f"{opts.experiment_name}_step_{opts.step}.pth"),
                        model, optimizer, scheduler, epoch, best_miou
                    )

    # Final test
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        sampler=DistributedSampler(test_ds),
        num_workers=opts.num_workers
    )
    test_loss, test_metrics = trainer.validate(test_loader, metrics, logger)
    if is_main:
        logger.info(f"Final Test mIoU: {test_metrics['mean_iou']:.4f}")
        if WANDB_AVAILABLE and opts.use_wandb:
            wandb.log({"test_mIoU": test_metrics['mean_iou']})

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_parser()
    opts = parser.parse_args()

    os.makedirs(opts.log_dir, exist_ok=True)
    os.makedirs(opts.ckpt_dir, exist_ok=True)

    main(opts)