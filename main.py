import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os

from dataset import EnhancedTimeSeriesDataset
from model import FixedFocusClassTransformer
from trainer import FixedFocusClassTrainer
from data_utils import prepare_datasets
from utils import create_sample_data

def fixed_main():
    sequence_length = 8
    prediction_length = 52
    batch_size = 32
    epochs = 200
    focus_classes = [1]
    data_dir = 'data'
    model_dir = 'models'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        excel_file = "AI_1_modified.xlsx"
        train_df, val_df = prepare_datasets(excel_file, data_dir=data_dir, force_split=False)
    except FileNotFoundError:
        print(f"文件 {excel_file} 未找到，创建示例数据...")
        excel_file = create_sample_data()
        train_df, val_df = prepare_datasets(excel_file, data_dir=data_dir, force_split=True)

    print("\n创建训练数据集...")
    train_dataset = EnhancedTimeSeriesDataset(
        train_df, sequence_length, prediction_length,
        augment_data=True, focus_classes=focus_classes
    )

    train_scaler = train_dataset.get_scaler()
    train_label_encoder = train_dataset.get_label_encoder()

    print("\n创建验证数据集...")
    val_dataset = EnhancedTimeSeriesDataset(
        val_df, sequence_length, prediction_length,
        augment_data=False, focus_classes=focus_classes,
        scaler=train_scaler, label_encoder=train_label_encoder
    )

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  验证集: {len(val_dataset)} 个样本")

    train_targets = train_dataset.encoded_labels
    class_weights = train_dataset.class_weights
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    os.makedirs(model_dir, exist_ok=True)

    model = FixedFocusClassTransformer(
        input_dim=train_dataset.feature_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        num_classes=train_dataset.num_classes,
        prediction_length=prediction_length,
        dropout=0.2,
        focus_classes=focus_classes
    ).to(device)

    trainer = FixedFocusClassTrainer(
        model, device, model_dir=model_dir,
        class_weights=train_dataset.class_weights,
        initial_reg_weight=1.0, initial_cls_weight=1.0,
        uncertainty_weighting=True, focus_classes=focus_classes
    )

    history = trainer.train(train_loader, val_loader, epochs, lr=0.0001)

    return model, train_dataset, history, device


if __name__ == "__main__":
    model, train_dataset, history, device = fixed_main()