"""
train_model.py
Standalone training script for the AI Decision Engine.
"""
import ml_engine

if __name__ == "__main__":
    print("Training attention-profile classifier on synthetic screening data...")
    accuracy = ml_engine.train_and_save_model()
    print(f"Done. Holdout test accuracy: {accuracy * 100:.1f}%")
    print(f"Saved model  -> {ml_engine.MODEL_PATH}")
    print(f"Saved scaler -> {ml_engine.SCALER_PATH}")
