import time
import torch
from tqdm import tqdm
from .losses import dice_coefficient, iou_score


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Args:
        patience (int): How many epochs to wait for improvement before stopping
        min_delta (float): Minimum change to qualify as improvement
        restore_best_weights (bool): Whether to restore model weights from best iteration
    """

    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = self.counter
        else:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

    def restore_weights(self, model):
        """
        Restore model to best weights
        """
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        epoch: Current epoch number

    Returns:
        dict: Dictionary with training metrics
    """
    model.train()

    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [TRAIN]")

    for images, masks in pbar:
        # Move data to device
        images = images.to(device)
        masks = masks.to(device).float().unsqueeze(1)  # Add channel dimension

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate metrics
        batch_dice = dice_coefficient(outputs, masks)
        batch_iou = iou_score(outputs, masks)

        # Update epoch metrics
        epoch_loss += loss.item()
        epoch_dice += batch_dice
        epoch_iou += batch_iou

        # Update progress bar
        pbar.set_postfix(loss=loss.item(), dice=batch_dice, iou=batch_iou)

    # Calculate average metrics
    num_batches = len(dataloader)
    epoch_loss /= num_batches
    epoch_dice /= num_batches
    epoch_iou /= num_batches

    return {
        'loss': epoch_loss,
        'dice': epoch_dice,
        'iou': epoch_iou
    }


def validate(model, dataloader, criterion, device):
    """
    Validate the model.

    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run validation on

    Returns:
        dict: Dictionary with validation metrics
    """
    model.eval()

    val_loss = 0
    val_dice = 0
    val_iou = 0

    # No gradient calculation during validation
    with torch.no_grad():
        # Progress bar
        pbar = tqdm(dataloader, desc="Validation")

        for images, masks in pbar:
            # Move data to device
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1)  # Add channel dimension

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Calculate metrics
            batch_dice = dice_coefficient(outputs, masks)
            batch_iou = iou_score(outputs, masks)

            # Update validation metrics
            val_loss += loss.item()
            val_dice += batch_dice
            val_iou += batch_iou

            # Update progress bar
            pbar.set_postfix(loss=loss.item(), dice=batch_dice, iou=batch_iou)

    # Calculate average metrics
    num_batches = len(dataloader)
    val_loss /= num_batches
    val_dice /= num_batches
    val_iou /= num_batches

    return {
        'val_loss': val_loss,
        'val_dice': val_dice,
        'val_iou': val_iou
    }


def train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scheduler=None,
        num_epochs=50,
        device="cuda",
        patience=10,
        checkpoint_path=None
):
    """
    Train and validate model.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to run training on
        patience: Patience for early stopping
        checkpoint_path: Path to save best model weights

    Returns:
        dict: Dictionary with training history
    """
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience)

    # Initialize history dictionary
    history = {
        'loss': [], 'dice': [], 'iou': [],
        'val_loss': [], 'val_dice': [], 'val_iou': [],
        'lr': []
    }

    # Training loop
    start_time = time.time()

    print(f"Training on {device}")
    model = model.to(device)

    for epoch in range(num_epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)

        # Update learning rate if scheduler exists
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()

        # Update history
        history['loss'].append(train_metrics['loss'])
        history['dice'].append(train_metrics['dice'])
        history['iou'].append(train_metrics['iou'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_dice'].append(val_metrics['val_dice'])
        history['val_iou'].append(val_metrics['val_iou'])
        history['lr'].append(current_lr)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(
            f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(
            f"Val Loss: {val_metrics['val_loss']:.4f}, Dice: {val_metrics['val_dice']:.4f}, IoU: {val_metrics['val_iou']:.4f}")
        print(f"Learning Rate: {current_lr:.8f}")

        # Check early stopping
        early_stopping(val_metrics['val_dice'], model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Save best model
        if checkpoint_path is not None and val_metrics['val_dice'] == early_stopping.best_score:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    # Restore best weights
    early_stopping.restore_weights(model)

    # Print training summary
    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes")
    print(f"Best validation Dice: {early_stopping.best_score:.4f}")

    return history