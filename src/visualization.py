import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


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



def plot_predictions_with_raw_images(model, dataset, dataset_df, device, num_samples=5, threshold=0.5):
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get processed sample for prediction
            image, mask = dataset[idx]

            # Convert to batch format and move to device
            image_tensor = image.unsqueeze(0).to(device)

            # Get prediction
            output = model(image_tensor)
            output = torch.sigmoid(output)

            # Load the raw image directly for visualization
            img_path = dataset_df.iloc[idx]['sat_image_path']
            mask_path = dataset_df.iloc[idx]['mask_path']

            # Load raw image and mask
            raw_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            raw_mask = (raw_mask > 128).astype(np.uint8)

            # Get prediction as numpy
            pred_np = output.squeeze().detach().cpu().numpy()
            pred_binary = (pred_np > threshold).astype(np.uint8)

            # Plot
            axes[i, 0].imshow(raw_image)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(raw_mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_binary, cmap='gray')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
