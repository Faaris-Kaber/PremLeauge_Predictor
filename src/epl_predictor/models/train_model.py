"""
neural network training module

trains a pytorch model to predict premier league match outcomes using
engineered features. handles class imbalance, uses chronological splits,
supports gpu (cuda/xpu) or cpu, with early stopping.
"""

import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from epl_predictor import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """auto-detect best available device for training."""
    if torch.cuda.is_available():
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("Using Intel GPU (XPU)")
        return torch.device('xpu')
    else:
        print("Using CPU")
        return torch.device('cpu')

device = get_device()


class EPLNet(nn.Module):
    """simple feed-forward net: input -> 64 -> 32 -> 3 classes."""
    
    def __init__(self, input_dim: int, dropout_rate: float = 0.3) -> None:
        super(EPLNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EPLPredictor:
    """handles training and evaluation of the match prediction model."""
    
    def __init__(self) -> None:
        self.model: EPLNet | None = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: list[str] | None = None
    
    def load_features(self) -> pd.DataFrame | None:
        """load the engineered features dataset."""
        input_path = config.PROCESSED_DATA_DIR / "features_dataset.csv"
        
        if not input_path.exists():
            logger.error("Features file not found. Run build_features.py first.")
            return None
        
        logger.info(f"Loading features from: {input_path}")
        df = pd.read_csv(input_path, parse_dates=['Date'])
        logger.info(f"Loaded {df.shape[0]} matches with {df.shape[1]} columns")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """prep data for training, returns features, targets, and dates."""
        logger.info("Preparing data...")
        
        # target encoding: H=0, D=1, A=2
        target_map = {'H': 0, 'D': 1, 'A': 2}
        df['Target'] = df['FTR'].map(target_map)
        df = df.dropna(subset=['Target'])
        
        # select engineered features
        feature_cols = [col for col in df.columns if 'RollingAvg' in col or '_Diff' in col]
        df = df.dropna(subset=feature_cols)
        
        logger.info(f"Using {len(feature_cols)} features, {df.shape[0]} matches")
        
        # class distribution
        counts = df['FTR'].value_counts()
        logger.info(f"Classes: H={counts.get('H', 0)}, D={counts.get('D', 0)}, A={counts.get('A', 0)}")
        
        self.feature_columns = feature_cols
        return df[feature_cols], df['Target'], df['Date']
    
    def split_chronologically(
        self, X: pd.DataFrame, y: pd.Series, dates: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """split data by date to prevent temporal leakage."""
        logger.info("Creating chronological splits...")
        
        X_arr = X.values
        y_arr = y.values
        dates = pd.to_datetime(dates)
        
        train_cutoff = pd.Timestamp('2023-05-31')
        val_cutoff = pd.Timestamp('2024-05-31')
        
        train_mask = dates <= train_cutoff
        val_mask = (dates > train_cutoff) & (dates <= val_cutoff)
        test_mask = dates > val_cutoff
        
        logger.info(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        
        return (
            X_arr[train_mask], X_arr[val_mask], X_arr[test_mask],
            y_arr[train_mask], y_arr[val_mask], y_arr[test_mask]
        )
    
    def compute_class_weights(self, y_train: np.ndarray) -> torch.Tensor:
        """compute class weights to handle imbalance."""
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        labels = {0: 'Home', 1: 'Draw', 2: 'Away'}
        for cls, weight in zip(classes, weights):
            logger.info(f"  {labels[int(cls)]}: {weight:.3f}x")
        
        return torch.tensor(weights, dtype=torch.float32).to(device)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        dropout: float = 0.3
    ) -> None:
        """train the model with early stopping and class weights."""
        logger.info(f"Starting training on {device}...")
        
        # scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val_scaled).to(device)
        y_val_t = torch.LongTensor(y_val).to(device)
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
        )
        
        # model
        self.model = EPLNet(X_train.shape[1], dropout_rate=dropout).to(device)
        
        # loss with class weights
        logger.info("Computing class weights...")
        class_weights = self.compute_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
        )
        
        # training loop
        logger.info(f"Training for up to {epochs} epochs...")
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # train
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            
            # validate
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_acc = (val_outputs.argmax(dim=1) == y_val_t).float().mean().item()
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(best_model_state)
        logger.info("Training complete!")
    
    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """evaluate model on test set."""
        logger.info("Evaluating on test set...")
        
        X_test_scaled = self.scaler.transform(X_test)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_t)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # metrics
        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average='macro')
        f1_per_class = f1_score(y_test, predictions, average=None)
        
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        
        target_names = ['Home Win', 'Draw', 'Away Win']
        print("\nPer-Class F1:")
        for name, f1, cls in zip(target_names, f1_per_class, [0, 1, 2]):
            print(f"  {name}: {f1:.3f} (n={np.sum(y_test == cls)})")
        
        # confusion matrix
        cm = confusion_matrix(y_test, predictions)
        print("\nConfusion Matrix:")
        print(f"{'':12} | {'H':>5} | {'D':>5} | {'A':>5}")
        print("-" * 40)
        for name, row in zip(target_names, cm):
            print(f"{name:12} | {row[0]:>5} | {row[1]:>5} | {row[2]:>5}")
        
        print("="*60)
        
        return accuracy, predictions, probabilities
    
    def save_model(self, filename: str = 'epl_predictor.pth') -> None:
        """save trained model to disk."""
        save_path = config.PROJECT_ROOT / 'saved_models' / filename
        save_path.parent.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, save_path)
        
        logger.info(f"Model saved to: {save_path}")


def main() -> None:
    """main training pipeline."""
    print("="*60)
    print("EPL MATCH PREDICTOR - NEURAL NETWORK TRAINING")
    print("="*60)
    
    predictor = EPLPredictor()
    
    df = predictor.load_features()
    if df is None:
        return
    
    X, y, dates = predictor.prepare_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_chronologically(X, y, dates)
    
    predictor.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001, dropout=0.3)
    
    accuracy, _, _ = predictor.evaluate(X_test, y_test)
    predictor.save_model()
    
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print("Model saved to: saved_models/epl_predictor.pth")


if __name__ == "__main__":
    main()
