from utils import setup_logger, log_config, log_metrics, close_logger

# Setup
rank = 0  # From distributed init
logger = setup_logger(
    log_dir="./logs/stage_1",
    rank=rank,
    debug=False,
    use_wandb=True,
    project_name="gcoe-css-voc"
)

# Log config
config = {
    "dataset": "voc",
    "task": "15-1",
    "epochs": 80,
    "lr": 1e-3,
    "batch_size": 4
}
log_config(logger, config, rank=rank)

# During training
metrics = {"mIoU": 0.72, "loss": 0.45}
log_metrics(logger, metrics, step=10, prefix="val/", rank=rank)

# At the end
close_logger(logger, rank=rank)