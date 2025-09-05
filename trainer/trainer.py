trainer = GCoETrainer(
    model=model,
    model_old=model_old,
    device=device,
    num_classes=21,
    old_classes=15,
    lambda_kd=1.0,
    is_main=(rank == 0)
)

# Train
trainer.dual_phase_training(train_loader, val_loader, epochs_phase1=80, epochs_phase2=20)

# Update memory
trainer.select_and_enhance_memory(train_loader, num_samples=100)

# Compute metrics
metrics = trainer.compute_metrics(old_feats, new_feats, psi_scores, iou_b, iou_a)
print(metrics)