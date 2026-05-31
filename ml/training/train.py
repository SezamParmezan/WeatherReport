import mlflow

import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import TimeSeriesDataSet
from pathlib import Path

from ml.data.dataset import load_and_process_data, split_data, make_timeseries_dataset, make_val_test_datasets
from ml.data.features import build_all_features

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data/processed"


"""This script is responsible for training a Temporal Fusion Transformer model for time series forecasting
    It includes the following steps:
    1. Data Preparation: It loads the processed data, builds features, splits the data into train, validation and test sets, and creates TimeSeriesDataSet objects for training and validation
    2. Model Building: It builds a Temporal Fusion Transformer model for time series forecasting.
    3. Trainer Building: It builds a PyTorch Lightning Trainer with early stopping and model checkpointing callbacks
    4. Training: It trains the model using the trainer and logs the best validation loss and the trained model to MLflow

Note: The script assumes that the processed data is stored in the "data/processed" directory and that the necessary features have been built using the build_all_features function
"""


def prepare_data() -> tuple[TimeSeriesDataSet, pl.LightningDataModule]:
    '''
    It prepares the data for training a PyTorch Forecasting model
    It loads the processed data, builds features, splits the data into train, 
    validation and test sets, and creates TimeSeriesDataSet objects for training and validation.
    '''

    files = sorted(PROCESSED_DIR.glob("rome_*.parquet"))
    
    df = load_and_process_data(PROCESSED_DIR)
    df = build_all_features(df)
    df = df.dropna()
    train_df, val_df, test_df = split_data(df)

    train_dataset = make_timeseries_dataset(train_df, encoder_len=72, prediction_len=24)
    val_dataset, test_dataset = make_val_test_datasets(train_dataset, val_df, test_df)

    train_loader = train_dataset.to_dataloader(batch_size=64, shuffle=False, num_workers=0)
    val_loader = val_dataset.to_dataloader(batch_size=64, shuffle=False, num_workers=0)

    return train_dataset, train_loader, val_loader


def build_model(train_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    '''
    It builds a Temporal Fusion Transformer model for time series forecasting
    '''

    return TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=3e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        loss=QuantileLoss(),
        log_interval=10,
    )


def build_trainer():
    """
    It builds a PyTorch Lightning Trainer with early stopping and model checkpointing callbacks
        - EarlyStopping: It monitors the validation loss and stops training if it doesn't improve for 5 consecutive epochs.
        - ModelCheckpoint: It saves the model checkpoint with the lowest validation loss during training.
    """

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    checkpoint = ModelCheckpoint(
        dirpath=Path(__file__).parent.parent / "models",
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    return pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        callbacks=[early_stop, checkpoint],
        gradient_clip_val=0.1,
    ), checkpoint


if __name__ == "__main__":
    train_dataset, train_loader, val_loader = prepare_data()
    model = build_model(train_dataset)
    trainer, checkpoint_cb = build_trainer()

    with mlflow.start_run():
        mlflow.log_params({
            "encoder_len": 72,
            "prediction_len": 24,
            "hidden_size": 64,
            "dropout": 0.1,
            "learning_rate": 3e-3,
        })

        trainer.fit(model, train_loader, val_loader)

        mlflow.log_metric("best_val_loss", checkpoint_cb.best_model_score.item())
        mlflow.pytorch.log_model(model, "model")