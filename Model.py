# Be sure to modify the Metadatas and CustomDataset class to match your dataset annotation format.

#METADATA
BATCHSIZE = 1
NUM_CLASSES = 2 
EPOCH_NUM = 100
SHUFFLE = True
LR = 0.0001 
DECAY = 0.0005 
CONTINUE = False # If its not True the model wont continue from checkpoint
CHECKPOINT_PATH = "/path/to/checkpoint_folder"
train_path = "/path/to/train_dataset"
val_path = "/path/to/val_dataset"

import os
import json
import torch
import numpy as np
import cv2
import timm
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import MaskRCNN
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou
from torch.amp import autocast, GradScaler



class Model():    """
    Model class for training and validating a Mask R-CNN model with a custom backbone and dataset.
    The Model initializes with ConvNeXt-base as the backbone and MaskRCNN as the model. You may customize the Model by your choice.
    The optimizer also initializes as AdamW but can be changed to any other optimizer.

    IMPORTANT: You must modify the CustomDataset class to match your dataset annotation format.
    
    The dataset hierarchy of mine is:
        Dataset
        ├── train
        │   ├── train.json
        │   ├── image_001.jpg
        │   ├── image_002.jpg
        │   └── ...
        │
        └── valid
            ├── valid.json
            ├── image_001.jpg
            ├── image_002.jpg
            └── ...

    Attributes:
        device (str): Device to run the model on ("cuda" or "cpu").
        trainset (DataLoader): DataLoader for the training dataset.
        validset (DataLoader): DataLoader for the validation dataset.
        _model (nn.Module): The Mask R-CNN model.
        optimizer (torch.optim.Optimizer): Optimizer for training.


    Methods:
        __init__(self, device="cuda", optimizer=None, train_path="./train", val_path="./valid"):
            Initializes the Model, prepares datasets, creates the model, and sets up the optimizer. 

        _prepareDataset(self, train_path, val_path):
            Loads annotations, applies transforms, and returns DataLoaders for training and validation.

        createModel(self, num_classes=NUM_CLASSES, featExtr=..., return_layers=..., in_channels_list=..., out_channels=256):
            Creates and returns a Mask R-CNN model with a custom backbone and FPN. To modify the model, you can call the createModel() function with the required parameters.

        _calculate_iou(self, outputs, targets):
            Calculates the Intersection over Union (IoU) matrix between predicted and target bounding boxes.

        _calculate_ap(self, outputs, targets):
            Calculates the mean average precision (mAP) for the given outputs and targets.

        _train_batch(self, images, targets, mode, scaler):
            Performs a single training or validation step on a batch, computes losses, and updates the model if in training mode.

        _train_epoch(self, dataloader, mode, scaler, epoch):
            Trains or validates the model for one epoch, accumulates and returns average losses.

        _valid_metrics(self, dataloader):
            Evaluates the model on the validation set, returning average IoU and mAP metrics.

        _load_checkpoint(self, path):
            Loads model and optimizer state from a checkpoint file.

        _save_checkpoint(self, save_path, epoch):
            Saves the current model and optimizer state to a checkpoint file.

        _save_logs(self, log_path, epoch, train_loss, valid_loss, map, avg_iou):
            Appends training and validation metrics to a log file.

        train(self, epoch_num, checkpoint=None):
            Runs the training loop for a specified number of epochs, periodically validating and saving checkpoints/logs.

    Example usage:
        model = Model(device="cuda", train_path=train_path, val_path=val_path)
        model.train(epoch_num=50)

    IMPORTANT:
        You must modify the CustomDataset class to match your dataset annotation format.

    Example annotation:
        [
              {
    "image_id": 15,
    "file_name": "image_015.jpg",
    "width": 512,
    "height": 512,
    "bbox": [
      419,
      0,
      511,
      32
    ],
    "segmentation": [
      [
        420.1771,
        3.5605,
        420.7582,
        14.242,
        419.3053,
      ]
    ]
  },
  ...
  Each line in the annotation file corresponds to one object instance at the given image_id.

    """
    def __init__(self, 
                 device="cuda",
                 optimizer=None,
                 train_path="./train", 
                 val_path="./valid"):
        
        self.device = device
        (self.trainset, self.validset) = self._prepareDataset(train_path, val_path)
        self.createModel()

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self._model.parameters(), lr=LR, weight_decay=DECAY)
        else:
            self.optimizer = optimizer

        print("Model initialized successfully.")

    def _prepareDataset(self, train_path, val_path):
        # prepares the dataset using CustomDataset class and returns dataloader_train and dataloader_valid in a tuple

        with open(os.path.join(train_path, "train.json")) as f:
            train_ann = json.load(f)
        with open(os.path.join(val_path, "valid.json")) as f:
            valid_ann = json.load(f)

        my_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset_train = CustomDataset(train_path, train_ann, transform=my_transform)
        dataset_valid = CustomDataset(val_path, valid_ann, transform=my_transform)

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=BATCHSIZE,
            shuffle=SHUFFLE,
            collate_fn=lambda batch: tuple(zip(*batch))
        )

        dataloader_valid = DataLoader(
            dataset_valid,
            batch_size=BATCHSIZE,
            shuffle=False,
            collate_fn=lambda batch: tuple(zip(*batch))
        )

        return (dataloader_train, dataloader_valid)

    def createModel(self, 
                     num_classes = NUM_CLASSES, 
                     featExtr = timm.create_model('convnext_base', pretrained=True, features_only=True), 
                     return_layers = {
                         "stages_0": "0",
                         "stages_1": "1",
                         "stages_2": "2",
                         "stages_3": "3"
                     }, 
                     in_channels_list = [128, 256, 512, 1024], 
                     out_channels = 256):
        # Creates model with default parameters if not provided


        backbone = BackboneWithFPN(
            backbone = featExtr,
            return_layers = return_layers,
            in_channels_list = in_channels_list,
            out_channels = out_channels)

        model = MaskRCNN(backbone, num_classes=num_classes)

        self._model = model


    def _calculate_iou(self, outputs, targets):
        # Calculates the Intersection over Union (IoU) between predicted and target boxes

        boxes1 = outputs
        boxes2 = targets

        if not isinstance(boxes1, torch.Tensor) or not isinstance(boxes2, torch.Tensor):
            return torch.tensor([0.0])

        iou_matrix = box_iou(boxes1, boxes2)
        return iou_matrix

    def _calculate_ap(self, outputs, targets):
        # Calculates the Average Precision (AP) between predicted and target boxes

        for output in outputs:
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    output[k] = v.to(self.device)

        for target in targets:
            for k, v in target.items():
                if isinstance(v, torch.Tensor):
                    target[k] = v.to(self.device)

        metric = MeanAveragePrecision()
        metric.update(outputs, targets)
        return metric.compute()



    def _train_batch(self, images, targets, mode, scaler):
        # Trains the model on a single batch of images and targets

        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        if mode == "train":
            self.optimizer.zero_grad()
        elif mode == "valid":
            with torch.no_grad():
                with autocast(device_type='cuda', enabled=True):
                    results = self._model(images, targets)
                    total_loss = sum(loss for loss in results.values())



        with autocast(device_type='cuda', enabled=True):
            results = self._model(images, targets)
            total_loss = sum(loss for loss in results.values())


        loss_classifier = results.get('loss_classifier', torch.tensor(0.0, device=self.device))
        loss_box_reg = results.get('loss_box_reg', torch.tensor(0.0, device=self.device))
        loss_mask = results.get('loss_mask', torch.tensor(0.0, device=self.device))
        loss_objectness = results.get('loss_objectness', torch.tensor(0.0, device=self.device))
        loss_rpn_box_reg = results.get('loss_rpn_box_reg', torch.tensor(0.0, device=self.device))
        
        
        if mode == "train":

            scaler.scale(total_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

        loss_dict = {}
        loss_dict["total"] = total_loss.item()
        loss_dict["classifier"] = loss_classifier.item()
        loss_dict["box"] = loss_box_reg.item()
        loss_dict["mask"] = loss_mask.item()
        loss_dict["objectness"] = loss_objectness.item()
        loss_dict["rpn_box"] = loss_rpn_box_reg.item()  

        return loss_dict


    def _train_epoch(self, dataloader, mode, scaler, epoch):
        # Trains the model for one epoch

        torch.cuda.empty_cache()
        self._model.train()

        total_loss_dict = { "total": [], "classifier": [], "box": [], "mask": [], "objectness": [], "rpn_box": [] }

        if mode == "train":
            desc_str = f"Training (Epoch {epoch + 1})" if epoch is not None else "Training"
        else:
            desc_str = "Validating"
        progress_bar = tqdm(dataloader, desc=desc_str, leave=None)

        for images, targets in progress_bar:
            loss_dict = self._train_batch(images, targets, mode=mode, scaler=scaler)

            progress_bar.set_postfix({
                'Total': f"{loss_dict["total"]:.4f}",
                'Cls': f"{loss_dict["classifier"]:.4f}",
                'Box': f"{loss_dict["box"]:.4f}",
                'Mask': f"{loss_dict["mask"]:.4f}",
                'Obj': f"{loss_dict["objectness"]:.4f}",
                'RPN': f"{loss_dict["rpn_box"]:.4f}"
            })

            total_loss_dict["total"].append(loss_dict["total"])
            total_loss_dict["classifier"].append(loss_dict["classifier"])
            total_loss_dict["box"].append(loss_dict["box"])
            total_loss_dict["mask"].append(loss_dict["mask"])
            total_loss_dict["objectness"].append(loss_dict["objectness"])
            total_loss_dict["rpn_box"].append(loss_dict["rpn_box"])

        print(f"\nEpoch Summary:")
        print(f"  Avg Total Loss: {np.mean(total_loss_dict['total']):.4f}")
        print(f"  Avg Classifier Loss: {np.mean(total_loss_dict['classifier']):.4f}")
        print(f"  Avg Box Loss: {np.mean(total_loss_dict['box']):.4f}")
        print(f"  Avg Mask Loss: {np.mean(total_loss_dict['mask']):.4f}")
        print(f"  Avg Objectness Loss: {np.mean(total_loss_dict['objectness']):.4f}")
        print(f"  Avg RPN Box Loss: {np.mean(total_loss_dict['rpn_box']):.4f}")


        mean_losses = {k: np.mean(v) for k, v in total_loss_dict.items()}
        return mean_losses
    
    def _valid_metrics(self, dataloader):
        # Calculates the validation metrics

        self._model.eval()
        iou_list = []
        ap_dicts = []
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self._model(images)
                for output, target in zip(outputs, targets):
                    iou_matrix = self._calculate_iou(output['boxes'], target['boxes'])
                    if iou_matrix.numel() > 0:
                        mean_iou = iou_matrix.max(dim=1)[0].mean().item() 
                    else:
                        mean_iou = 0.0
                    iou_list.append(mean_iou)
                    ap = self._calculate_ap([output], [target])
                    ap_cpu = {
                        k: (v.cpu().item() if torch.is_tensor(v) and v.numel() == 1 else v.cpu().tolist() if torch.is_tensor(v) else v)
                        for k, v in ap.items()}
                    ap_dicts.append(ap_cpu)

        if ap_dicts:
            keys = ap_dicts[0].keys()
            mean_ap = {}
            for k in keys:
                vals = [ap[k] if isinstance(ap[k], (float, int)) and ap[k] >= 0 and not np.isnan(ap[k]) else 0.0 for ap in ap_dicts]
                mean_ap[k] = np.mean(vals)
        else:
            mean_ap = {}
        return np.mean(iou_list), mean_ap

    def _load_checkpoint(self, path):
        # Loads the model and optimizer state from a checkpoint

        if CONTINUE is False:
            print("Training from scratch")
            return 0
        
        if os.path.exists(path):
            print(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:

                self._model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                else: start_epoch = 0
                print(f"Resumed from epoch {start_epoch}")

            else:
                self._model.load_state_dict(checkpoint)
                print("Loaded model weights from checkpoint")

            return start_epoch
    
    def _save_checkpoint(self, save_path, epoch):
        # Saves the model and optimizer state to a checkpoint

        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }
        os.makedirs(save_path, exist_ok=True)
        torch.save(checkpoint, os.path.join(save_path, f"epoch_{epoch}.pth"))
        print(f"Checkpoint saved at {save_path} as epoch_{epoch}.pth")

    def _save_logs(self, log_path, epoch, train_loss, valid_loss, map, avg_iou,):
        # Saves training logs to a JSON file

        log_data = {
            "epoch": epoch + 1,
            "train_loss": round(float(train_loss["total"]), 4),
            "val_loss": round(float(valid_loss["total"]), 4),
            "mAP" : round(float(map['map']), 4),
            "mAP@50": round(float(map['map_50']), 4),
            "iou": round(float(avg_iou), 4),
        }

        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "training_log.json"), "a") as log_file:
            log_file.write(json.dumps(log_data) + ",\n")
        

    def train(self, epoch_num, checkpoint=None):
        # Runs the training loop for the specified number of epochs, handles checkpointing, and performs validation.

        if checkpoint is not None:
            if os.path.exists(checkpoint):
                start_epoch = self._load_checkpoint(checkpoint)
        else: 
            start_epoch = 0

        self._model.to(self.device)
        scaler = GradScaler()

        for curr_epoch in range(start_epoch, epoch_num):
            print(f"Epoch:{curr_epoch+1}/{epoch_num}")
            train_losses = self._train_epoch(self.trainset, mode="train", scaler=scaler, epoch=curr_epoch)

            if (curr_epoch + 1) % 5 == 0:
                valid_losses = self._train_epoch(self.validset, mode="valid", scaler=scaler, epoch=curr_epoch) # Model remains in train mode to ensure loss computation.
                avg_iou, mAP = self._valid_metrics(self.validset)
                print(f"Validation mAP: {mAP['map']:.4f}, mAP@50: {mAP['map_50']:.4f}, Avg IoU: {avg_iou:.4f}")
                self._save_logs("./logs", curr_epoch, train_losses, valid_losses, mAP, avg_iou)

            if (curr_epoch + 1) % 10 == 0:
                self._save_checkpoint("./checkpoints", curr_epoch + 1)






class CustomDataset(Dataset):
    def __init__(self, data_dir, anns, transform=None):
        self.data_dir = data_dir
        self.anns = anns
        self.transform = transform
        self.images = [x for x in os.listdir(data_dir) if x.endswith(".jpg")]

    def __len__(self):
        return len(self.images)
    @staticmethod
    def polygons_to_mask(polygons, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        img = cv2.imread(img_path)
        target = {"boxes": [], "labels": [], "masks": []}

        temp = []
        for ann in self.anns:
            if ann['file_name'] == self.images[idx]:
                temp.append(ann)

        for ann in temp:
            if ann["bbox"] is not None:
                target["boxes"].append(torch.tensor(ann["bbox"], dtype=torch.float32))
                target["labels"].append(torch.tensor(1, dtype=torch.int64))  
                target["masks"].append(torch.tensor(CustomDataset.polygons_to_mask(ann["segmentation"], ann["height"], ann["width"])))

        if self.transform:
            img = self.transform(img)

        if target["boxes"]:
            target["boxes"] = torch.stack(target["boxes"])
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        if target["labels"]:
            target["labels"] = torch.stack(target["labels"])
        else:
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        if target["masks"]:
            target["masks"] = torch.stack(target["masks"])
        else:
            target["masks"] = torch.zeros((0, img.shape[0], img.shape[1]), dtype=torch.uint8)

        return img, target




def main():
    model = Model(device="cuda", train_path=train_path, val_path=val_path)
    model.train(epoch_num=EPOCH_NUM)


if __name__ == "__main__":
    main()


