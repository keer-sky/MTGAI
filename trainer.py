import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import json

class FixedFocusClassTrainer:
    def __init__(self, model, device, model_dir, class_weights=None,
                 initial_reg_weight=1.0, initial_cls_weight=1.0,
                 uncertainty_weighting=True, focus_classes=None):
        self.model = model
        self.device = device
        self.model_dir = model_dir
        self.focus_classes = focus_classes

        os.makedirs(model_dir, exist_ok=True)
        self.regression_criterion = nn.SmoothL1Loss()
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
            self.classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.classification_criterion = nn.CrossEntropyLoss()

        self.uncertainty_weighting = uncertainty_weighting
        if uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(2))
        else:
            self.regression_weight = initial_reg_weight
            self.classification_weight = initial_cls_weight

        self.train_metrics = {
            'total_loss': [], 'reg_loss': [], 'cls_loss': [],
            'reg_mae': [], 'cls_accuracy': [], 'reg_weight': [], 'cls_weight': [],
            'focus_class_accuracy': []
        }
        self.val_metrics = {
            'total_loss': [], 'reg_loss': [], 'cls_loss': [],
            'reg_mae': [], 'cls_accuracy': [], 'focus_class_accuracy': []
        }

    def compute_metrics(self, regression_pred, regression_target,
                        classification_pred, classification_target):
        reg_mae = torch.mean(torch.abs(regression_pred - regression_target)).item()
        cls_pred = torch.argmax(classification_pred, dim=1)
        cls_accuracy = (cls_pred == classification_target).float().mean().item()
        focus_class_accuracy = None
        if self.focus_classes is not None:
            focus_mask = torch.isin(classification_target, torch.tensor(self.focus_classes).to(self.device))
            if focus_mask.any():
                focus_class_accuracy = (cls_pred[focus_mask] == classification_target[focus_mask]).float().mean().item()
        return reg_mae, cls_accuracy, focus_class_accuracy

    def compute_task_weights(self, regression_loss, classification_loss, epoch):
        if self.uncertainty_weighting:
            reg_weight = torch.exp(-self.log_vars[0])
            cls_weight = torch.exp(-self.log_vars[1])
            reg_loss = reg_weight * regression_loss + 0.5 * self.log_vars[0]
            cls_loss = cls_weight * classification_loss + 0.5 * self.log_vars[1]
            return reg_loss, cls_loss, reg_weight.item(), cls_weight.item()
        else:
            if epoch < 30:
                reg_weight = 0.5
                cls_weight = 1.5
            elif epoch < 80:
                reg_weight = 0.8
                cls_weight = 1.2
            else:
                reg_weight = 1.0
                cls_weight = 1.0
            return (reg_weight * regression_loss,
                    cls_weight * classification_loss,
                    reg_weight, cls_weight)

    def train(self, train_loader, val_loader, epochs, lr=0.001):
        if self.uncertainty_weighting:
            params = list(self.model.parameters()) + [self.log_vars]
        else:
            params = self.model.parameters()

        optimizer = optim.AdamW(params, lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        best_val_loss = float('inf')
        patience = 25
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_total_loss = 0
            train_reg_loss = 0
            train_cls_loss = 0
            train_reg_mae = 0
            train_cls_accuracy = 0
            train_focus_accuracy = 0
            avg_reg_weight = 0
            avg_cls_weight = 0
            focus_batch_count = 0

            for batch_idx, (data, regression_target, classification_target) in enumerate(train_loader):
                data = data.to(self.device)
                regression_target = regression_target.to(self.device)
                classification_target = classification_target.squeeze().to(self.device)

                optimizer.zero_grad()
                regression_pred, classification_pred = self.model(data)

                regression_loss = self.regression_criterion(regression_pred, regression_target)
                classification_loss = self.classification_criterion(classification_pred, classification_target)

                weighted_reg_loss, weighted_cls_loss, reg_weight, cls_weight = self.compute_task_weights(
                    regression_loss, classification_loss, epoch
                )
                total_loss = weighted_reg_loss + weighted_cls_loss

                avg_reg_weight += reg_weight
                avg_cls_weight += cls_weight

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                reg_mae, cls_accuracy, focus_accuracy = self.compute_metrics(
                    regression_pred, regression_target, classification_pred, classification_target
                )

                train_total_loss += total_loss.item()
                train_reg_loss += regression_loss.item()
                train_cls_loss += classification_loss.item()
                train_reg_mae += reg_mae
                train_cls_accuracy += cls_accuracy

                if focus_accuracy is not None:
                    train_focus_accuracy += focus_accuracy
                    focus_batch_count += 1

            self.model.eval()
            val_total_loss = 0
            val_reg_loss = 0
            val_cls_loss = 0
            val_reg_mae = 0
            val_cls_accuracy = 0
            val_focus_accuracy = 0
            val_focus_batch_count = 0

            with torch.no_grad():
                for data, regression_target, classification_target in val_loader:
                    data = data.to(self.device)
                    regression_target = regression_target.to(self.device)
                    classification_target = classification_target.squeeze().to(self.device)

                    regression_pred, classification_pred = self.model(data)

                    regression_loss = self.regression_criterion(regression_pred, regression_target)
                    classification_loss = self.classification_criterion(classification_pred, classification_target)

                    total_loss = regression_loss + classification_loss

                    reg_mae, cls_accuracy, focus_accuracy = self.compute_metrics(
                        regression_pred, regression_target, classification_pred, classification_target
                    )

                    val_total_loss += total_loss.item()
                    val_reg_loss += regression_loss.item()
                    val_cls_loss += classification_loss.item()
                    val_reg_mae += reg_mae
                    val_cls_accuracy += cls_accuracy

                    if focus_accuracy is not None:
                        val_focus_accuracy += focus_accuracy
                        val_focus_batch_count += 1

            avg_train_loss = train_total_loss / len(train_loader)
            avg_val_loss = val_total_loss / len(val_loader)
            avg_train_reg = train_reg_loss / len(train_loader)
            avg_train_cls = train_cls_loss / len(train_loader)
            avg_val_reg = val_reg_loss / len(val_loader)
            avg_val_cls = val_cls_loss / len(val_loader)
            avg_train_reg_mae = train_reg_mae / len(train_loader)
            avg_train_cls_acc = train_cls_accuracy / len(train_loader)
            avg_val_reg_mae = val_reg_mae / len(val_loader)
            avg_val_cls_acc = val_cls_accuracy / len(val_loader)
            avg_reg_weight = avg_reg_weight / len(train_loader)
            avg_cls_weight = avg_cls_weight / len(train_loader)

            avg_train_focus_acc = train_focus_accuracy / focus_batch_count if focus_batch_count > 0 else 0
            avg_val_focus_acc = val_focus_accuracy / val_focus_batch_count if val_focus_batch_count > 0 else 0

            self.train_metrics['total_loss'].append(avg_train_loss)
            self.train_metrics['reg_loss'].append(avg_train_reg)
            self.train_metrics['cls_loss'].append(avg_train_cls)
            self.train_metrics['reg_mae'].append(avg_train_reg_mae)
            self.train_metrics['cls_accuracy'].append(avg_train_cls_acc)
            self.train_metrics['reg_weight'].append(avg_reg_weight)
            self.train_metrics['cls_weight'].append(avg_cls_weight)
            self.train_metrics['focus_class_accuracy'].append(avg_train_focus_acc)

            self.val_metrics['total_loss'].append(avg_val_loss)
            self.val_metrics['reg_loss'].append(avg_val_reg)
            self.val_metrics['cls_loss'].append(avg_val_cls)
            self.val_metrics['reg_mae'].append(avg_val_reg_mae)
            self.val_metrics['cls_accuracy'].append(avg_val_cls_acc)
            self.val_metrics['focus_class_accuracy'].append(avg_val_focus_acc)

            scheduler.step()

            epoch_model_path = os.path.join(self.model_dir, f'model_epoch_{epoch + 1:03d}.pth')
            torch.save(self.model.state_dict(), epoch_model_path)

            current_val_metric = avg_val_loss * (1 - avg_val_focus_acc * 0.5) if avg_val_focus_acc > 0 else avg_val_loss

            if current_val_metric < best_val_loss:
                best_val_loss = current_val_metric
                patience_counter = 0
                best_model_path = os.path.join(self.model_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}/{epochs}: ')
                print(f'  Train - Loss: {avg_train_loss:.4f} | Reg: {avg_train_reg:.4f} | Cls: {avg_train_cls:.4f} | MAE: {avg_train_reg_mae:.4f} | Acc: {avg_train_cls_acc:.4f}')
                if avg_train_focus_acc > 0:
                    print(f'          Focus Acc: {avg_train_focus_acc:.4f}')
                print(f'  Val   - Loss: {avg_val_loss:.4f} | Reg: {avg_val_reg:.4f} | Cls: {avg_val_cls:.4f} | MAE: {avg_val_reg_mae:.4f} | Acc: {avg_val_cls_acc:.4f}')
                if avg_val_focus_acc > 0:
                    print(f'          Focus Acc: {avg_val_focus_acc:.4f}')
                if self.uncertainty_weighting:
                    print(f'  Weights - Reg: {avg_reg_weight:.4f} | Cls: {avg_cls_weight:.4f}')
                print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
                print(f'  模型已保存到: {epoch_model_path}')

            if patience_counter >= patience:
                print(f"早停在 epoch {epoch}")
                break

        metrics_path = os.path.join(self.model_dir, 'training_metrics.json')
        metrics_data = {'train_metrics': self.train_metrics, 'val_metrics': self.val_metrics}
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        best_model_path = os.path.join(self.model_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return {'train_metrics': self.train_metrics, 'val_metrics': self.val_metrics}