def log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr):
    print(
        f"[{epoch:03d}] "
        f"LR {lr:.2e} | "
        f"Train {train_loss:.4f}/{train_acc:.4f} | "
        f"Val {val_loss:.4f}/{val_acc:.4f}"
    )
