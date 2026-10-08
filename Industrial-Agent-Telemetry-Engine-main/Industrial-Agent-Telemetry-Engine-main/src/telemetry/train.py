import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.telemetry.dataset import CMAPSSDataPipeline
from src.telemetry.model import MultiTaskAttentionNetwork

class DualTaskLoss(nn.Module):
    """Computes the combined loss for RUL regression and Sequence Reconstruction."""
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha  # Weight for RUL RMSE
        self.beta = beta    # Weight for Reconstruction MSE
        self.rul_criterion = nn.MSELoss()
        self.recon_criterion = nn.MSELoss()

    def forward(self, rul_pred, rul_target, recon_pred, x_input):
        # Calculate RMSE for RUL to keep it in the same unit domain (cycles)
        rul_loss = torch.sqrt(self.rul_criterion(rul_pred.squeeze(), rul_target))
        
        # Calculate standard MSE for the anomaly reconstruction
        recon_loss = self.recon_criterion(recon_pred, x_input)
        
        total_loss = (self.alpha * rul_loss) + (self.beta * recon_loss)
        return total_loss, rul_loss, recon_loss

def nasa_asymmetric_score(rul_pred, rul_target):
    """
    NASA CMAPSS official evaluation metric.
    Exponentially penalizes late predictions (d > 0) more severely than early predictions (d < 0).
    """
    d = rul_pred.squeeze() - rul_target
    score = torch.where(
        d < 0,
        torch.exp(-d / 13.0) - 1.0,
        torch.exp(d / 10.0) - 1.0
    )
    return torch.sum(score).item()

class TelemetryTrainer:
    """Execution loop for the Multi-Task Attention Network."""
    def __init__(self, model, train_loader, test_loader, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Adam optimizer with slight weight decay for L2 regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = DualTaskLoss()
        
        self.model_dir = Path("data/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss, total_rul, total_recon = 0.0, 0.0, 0.0

        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            rul_pred, recon_pred = self.model(x_batch)

            loss, rul_loss, recon_loss = self.criterion(rul_pred, y_batch, recon_pred, x_batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in the LSTM
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_rul += rul_loss.item()
            total_recon += recon_loss.item()

        batches = len(self.train_loader)
        return total_loss / batches, total_rul / batches, total_recon / batches

    def evaluate(self):
        self.model.eval()
        total_rmse, total_recon, total_nasa_score = 0.0, 0.0, 0.0

        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                
                rul_pred, recon_pred = self.model(x_batch)

                rmse = torch.sqrt(nn.functional.mse_loss(rul_pred.squeeze(), y_batch))
                recon_loss = nn.functional.mse_loss(recon_pred, x_batch)
                nasa_score = nasa_asymmetric_score(rul_pred, y_batch)

                total_rmse += rmse.item()
                total_recon += recon_loss.item()
                total_nasa_score += nasa_score

        batches = len(self.test_loader)
        return total_rmse / batches, total_recon / batches, total_nasa_score

    def save_model(self, filename="telemetry_model_v1.pt"):
        path = self.model_dir / filename
        torch.save(self.model.state_dict(), path)
        return path

if __name__ == "__main__":
    print("Initializing Dual-Loss Training Loop Verification...")
    try:
        # 1. Spin up the data pipeline
        pipeline = CMAPSSDataPipeline(sequence_length=30)
        train_loader, test_loader = pipeline.prepare_data(batch_size=64)
        
        # 2. Initialize the dual-head architecture
        model = MultiTaskAttentionNetwork(input_dim=24, seq_len=30)
        
        # 3. Initialize trainer (using CPU for reliable local verification)
        trainer = TelemetryTrainer(model, train_loader, test_loader, device='cpu')
        
        print("Executing 1 Verification Epoch (This may take a few seconds)...")
        train_loss, train_rul, train_recon = trainer.train_epoch()
        val_rmse, val_recon, val_nasa = trainer.evaluate()
        
        # 4. Verify state persistence
        model_path = trainer.save_model()
        
        print(f"[SUCCESS] Training loop executed and model weights saved to {model_path}")
        print(f"[METRICS] Train -> Total Loss: {train_loss:.4f} | RUL RMSE: {train_rul:.4f} | Recon MSE: {train_recon:.4f}")
        print(f"[METRICS] Eval  -> RUL RMSE: {val_rmse:.4f} | Recon MSE: {val_recon:.4f} | NASA Score: {val_nasa:.2f}")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")