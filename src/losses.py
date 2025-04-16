import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Formula: 1 - 2*|A∩B|/(|A|+|B|)
    where A and B are the prediction and target sets.

    Args:
        smooth (float): Smoothing factor to prevent division by zero
        eps (float): Small constant to prevent numeric instability
    """

    def __init__(self, smooth=1.0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits, targets):
        # Apply sigmoid to get predictions in [0,1] range
        probs = torch.sigmoid(logits)

        # Flatten tensors
        batch_size = probs.shape[0]
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        # Calculate intersection and union
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        # Calculate Dice coefficient and loss
        dice = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)
        dice_loss = 1 - dice.mean()

        return dice_loss


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice loss.

    Args:
        dice_weight (float): Weight for Dice loss component
        bce_weight (float): Weight for BCE loss component
    """

    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super(BCEDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        bce_loss = self.bce_loss(logits, targets)

        # Combine losses
        loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        return loss


def dice_coefficient(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice coefficient for validation.

    Args:
        predictions (torch.Tensor): Predicted logits
        targets (torch.Tensor): Target masks
        threshold (float): Threshold for binary prediction
        smooth (float): Smoothing factor

    Returns:
        float: Dice coefficient
    """
    with torch.no_grad():
        # Apply sigmoid and threshold to get binary predictions
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > threshold).float()

        # Flatten the tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()

        # Calculate Dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.item()


def iou_score(predictions, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate IoU for validation.

    Args:
        predictions (torch.Tensor): Predicted logits
        targets (torch.Tensor): Target masks
        threshold (float): Threshold for binary prediction
        smooth (float): Smoothing factor

    Returns:
        float: IoU score
    """
    with torch.no_grad():
        # Apply sigmoid and threshold to get binary predictions
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > threshold).float()

        # Flatten the tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection

        # Calculate IoU
        iou = (intersection + smooth) / (union + smooth)

    return iou.item()


# Factory function for creating loss functions
def get_loss_function(loss_name="bce_dice", **kwargs):
    """
    Factory function to create loss functions.

    Args:
        loss_name (str): Name of the loss function
        **kwargs: Additional arguments for the loss function

    Returns:
        nn.Module: Loss function
    """
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == "dice":
        return DiceLoss(**kwargs)
    elif loss_name == "bce_dice":
        return BCEDiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")