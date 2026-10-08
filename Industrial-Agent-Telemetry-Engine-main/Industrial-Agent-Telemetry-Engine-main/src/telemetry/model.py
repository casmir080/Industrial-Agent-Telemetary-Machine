import torch
import torch.nn as nn

class MultiTaskAttentionNetwork(nn.Module):
    """
    Dual-Head PyTorch model for simultaneous RUL forecasting and Anomaly Detection.
    Architecture: Bi-LSTM Encoder -> Self-Attention -> [RUL Regression Head, Reconstruction Head]
    """
    def __init__(self, input_dim: int = 24, seq_len: int = 30, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4):
        super(MultiTaskAttentionNetwork, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # 1. Shared Temporal Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 2. Self-Attention Layer (hidden_dim * 2 because of Bi-LSTM)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 3. Head A: Remaining Useful Life (RUL) Regressor
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Forces output to be strictly positive (RUL cannot be negative)
        )
        
        # 4. Head B: Anomaly Detection (Sequence Reconstruction Decoder)
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * input_dim)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            rul_pred: Tensor of shape (batch_size, 1)
            recon_pred: Tensor of shape (batch_size, seq_len, input_dim)
        """
        # Encode temporal features
        lstm_out, _ = self.encoder(x) # Shape: (batch, seq_len, hidden_dim * 2)
        
        # Apply Self-Attention to focus on critical degradation steps
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out) 
        
        # Global context pooling (Mean pool across the sequence length dimension)
        context_vector = torch.mean(attn_out, dim=1) # Shape: (batch, hidden_dim * 2)
        
        # Head A: Predict RUL
        rul_pred = self.rul_head(context_vector)
        
        # Head B: Reconstruct sequence for anomaly score calculation
        recon_flat = self.anomaly_head(context_vector)
        recon_pred = recon_flat.view(x.size(0), self.seq_len, self.input_dim)
        
        return rul_pred, recon_pred

if __name__ == "__main__":
    print("Initializing Multi-Task Attention Network Verification...")
    try:
        # Simulate the exact batch dimensions from the dataset.py output
        batch_size = 64
        seq_len = 30
        features = 24
        
        # Initialize model and dummy tensor
        model = MultiTaskAttentionNetwork(input_dim=features, seq_len=seq_len)
        dummy_x = torch.randn(batch_size, seq_len, features)
        
        # Execute forward pass
        rul_out, recon_out = model(dummy_x)
        
        print(f"[SUCCESS] Network forward pass executed cleanly.")
        print(f"[METRICS] Input shape: {dummy_x.shape}")
        print(f"[METRICS] RUL Head output shape: {rul_out.shape} (Expected: {batch_size}, 1)")
        print(f"[METRICS] Anomaly Head output shape: {recon_out.shape} (Expected: {batch_size}, {seq_len}, {features})")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")