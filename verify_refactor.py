
import sys
import os
import torch
# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.ssl_trainer import SSLTrainer, SSLBackboneTaskWrapper

def test_imports():
    print("Testing imports...")
    try:
        from scripts import train_ssl
        from engine.ssl_trainer import SSLTrainer, SSLBackboneTaskWrapper
        print("Imports successful.")
    except ImportError as e:
        print(f"Import failed: {e}")
        sys.exit(1)

def test_wrapper():
    print("Testing SSLBackboneTaskWrapper...")
    encoder = torch.nn.Linear(10, 5)
    
    class MockTask(torch.nn.Module):
        def transform(self, batch):
            # Return batch as is with wrapper
            from types import SimpleNamespace
            return SimpleNamespace(inputs=batch, metadata={})
        
        def uses_model_directly(self):
            return False
            
        def compute_loss(self, output, batch):
            batch.metadata["acc"] = 0.5
            return output.sum()
            
    task = MockTask()
    wrapper = SSLBackboneTaskWrapper(encoder, task)
    
    # Fake batch
    batch = torch.randn(2, 10)
    loss, metrics = wrapper(batch)
    print(f"Wrapper output: loss={loss.item()}, metrics={metrics}")

if __name__ == "__main__":
    test_imports()
    test_wrapper()
