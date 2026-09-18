import matplotlib.pyplot as plt
import numpy as np
import re
import os
from pathlib import Path

log_file="/storage/projects/vinkle/ez_compass_imaging/code/pavan_unet/checkpoint/checkpoint_2/train.log"
output_file="/storage/projects/vinkle/ez_compass_imaging/code/pavan_unet/curves/train_curves.png"

os.makedirs(Path(output_file).parent, exist_ok=True)


pattern=re.compile(
    "Epoch (?P<epoch>\d+)/(?P<total_epochs>\d+) \| train loss (?P<train_loss>[\d\.]+) \| val loss (?P<val_loss>[\d\.]+) \| val dice (?P<val_dice>[\d\.]+) \| val IoU (?P<val_iou>[\d\.]+) \| lr (?P<lr>[\d\.]+)"
)

epochs=[]
train_loss=[]
val_loss=[]
val_dice=[]
val_iou=[]


with open(log_file, "r") as f:
    for line in f:
        match=pattern.search(line)
        if match:
            epochs.append(int(match.group("epoch")))
            train_loss.append(float(match.group("train_loss")))
            val_loss.append(float(match.group("val_loss")))
            val_dice.append(float(match.group("val_dice")))
            val_iou.append(float(match.group("val_iou")))

epochs=np.array(epochs)
train_loss=np.array(train_loss)
val_loss=np.array(val_loss)
val_dice=np.array(val_dice)
val_iou=np.array(val_iou)

print(f"Total epochs: {len(epochs)}")

plt.subplot(1, 2, 1)
plt.title("Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Validation Metrics")
plt.xlabel("Epochs")
plt.ylabel("Metrics")
plt.plot(epochs, val_dice, label="Validation Dice")
plt.plot(epochs, val_iou, label="Validation IoU")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(output_file)


plt.show()