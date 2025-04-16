import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize(**images):
    """
    Plot images in one row

    Args:
        **images: Dictionary of name, image pairs
    """
    n_images = len(images)
    plt.figure(figsize=(6 * n_images, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace('_', ' ').title(), fontsize=20)

        # Handle different image types
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if image.ndim == 3 and image.shape[0] == 3:  # CHW format
            image = image.transpose(1, 2, 0)

        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):  # Mask or grayscale
            plt.imshow(image.squeeze(), cmap='gray')
        else:
            plt.imshow(image)
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plot training history metrics

    Args:
        history: Dictionary with training history
    """
    # Filter out learning rate from metrics to plot
    metrics = [k for k in history.keys() if k != 'lr']

    plt.figure(figsize=(16, 12))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Dice coefficient
    plt.subplot(2, 2, 2)
    plt.plot(history['dice'], label='Training Dice')
    plt.plot(history['val_dice'], label='Validation Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot IoU
    plt.subplot(2, 2, 3)
    plt.plot(history['iou'], label='Training IoU')
    plt.plot(history['val_iou'], label='Validation IoU')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_predictions(model, dataset, device, num_samples=5, threshold=0.5):
    """
    Plot sample predictions from the model
    """
    # Set model to evaluation mode
    model.eval()

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    # Get random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get sample
            image, mask = dataset[idx]

            # Convert to batch format and move to device
            image_tensor = image.unsqueeze(0).to(device)

            # Get prediction
            output = model(image_tensor)
            output = torch.sigmoid(output)

            # Convert to numpy
            if isinstance(image, torch.Tensor):
                image_np = image.detach().cpu().numpy()
                if image_np.shape[0] == 3:  # CHW format
                    image_np = image_np.transpose(1, 2, 0)
            else:
                image_np = image

            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = mask

            pred_np = output.squeeze().detach().cpu().numpy()
            pred_binary = (pred_np > threshold).astype(np.uint8)

            # Denormalize image
            # For ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            if image_np.max() <= 1.0:  # If image is normalized
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)

            # Plot
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title("Image")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_binary, cmap='gray')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_overlay_predictions(model, dataset, device, num_samples=5, threshold=0.5):
    """
    Plot sample predictions overlaid on the original image

    Args:
        model: PyTorch model
        dataset: Dataset to sample from
        device: Device to run inference on
        num_samples: Number of samples to visualize
        threshold: Threshold for binary prediction
    """
    # Set model to evaluation mode
    model.eval()

    # Get random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for idx in indices:
            # Get sample
            image, mask = dataset[idx]

            # Convert to batch format and move to device
            image_tensor = image.unsqueeze(0).to(device)

            # Get prediction
            output = model(image_tensor)
            output = torch.sigmoid(output)

            # Visualize
            visualize_prediction(
                image=image,
                gt_mask=mask,
                pred_mask=output.squeeze(),
                threshold=threshold
            )


def visualize_prediction(image, gt_mask, pred_mask, threshold=0.5):
    """
    Visualize image, ground truth, prediction, and overlay

    Args:
        image: Input image (C,H,W) or (H,W,C)
        gt_mask: Ground truth mask
        pred_mask: Predicted mask (before thresholding)
        threshold: Threshold for binary prediction
    """
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.shape[0] == 3:  # CHW format
            image = image.transpose(1, 2, 0)

    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy().squeeze()

    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy().squeeze()

    # Apply threshold to prediction
    pred_binary = (pred_mask > threshold).astype(np.uint8)

    # Denormalize image if it's normalized
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if image.max() <= 1.0:  # If image is normalized
        image = image * std + mean
        image = np.clip(image, 0, 1)

    # Create colored masks for visualization
    gt_colored = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    gt_colored[gt_mask == 1] = [1, 0, 0]  # Red for ground truth

    pred_colored = np.zeros((*pred_binary.shape, 3), dtype=np.float32)
    pred_colored[pred_binary == 1] = [0, 1, 0]  # Green for prediction

    # Create overlay image
    overlay = image.copy()
    overlay = np.where(gt_colored > 0, gt_colored * 0.7 + overlay * 0.3, overlay)
    overlay = np.where(pred_colored > 0, pred_colored * 0.7 + overlay * 0.3, overlay)

    # Visualize
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(gt_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(pred_binary, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout()
    plt.show()