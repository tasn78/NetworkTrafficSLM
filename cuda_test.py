import os
import sys
import xgboost as xgb
import numpy as np


def check_cuda_environment():
    print("=== CUDA Environment Check ===")

    # Check CUDA_PATH
    cuda_path = os.environ.get('CUDA_PATH')
    print(f"CUDA_PATH: {cuda_path}")

    # Check if path exists
    if cuda_path:
        print(f"CUDA_PATH exists on filesystem: {os.path.exists(cuda_path)}")
        print(f"Contents of CUDA_PATH:")
        try:
            print(os.listdir(cuda_path))
        except Exception as e:
            print(f"Error listing CUDA_PATH: {e}")

    # Check XGBoost version
    print(f"\nXGBoost version: {xgb.__version__}")

    # Try to detect CUDA toolkit
    print("\nTrying to create a small GPU test...")
    try:
        # Create small test data
        data = np.random.rand(10, 3)
        labels = np.random.randint(2, size=10)

        # Create DMatrix
        dtrain = xgb.DMatrix(data, label=labels)

        # Setup parameters
        param = {
            'tree_method': 'hist',
            'device': 'cuda',
            'gpu_id': 0
        }

        # Try training
        bst = xgb.train(param, dtrain, num_boost_round=1)
        print("GPU test successful!")

    except Exception as e:
        print(f"GPU test failed: {str(e)}")

    # Print system PATH
    print("\nSystem PATH entries:")
    for path in os.environ['PATH'].split(os.pathsep):
        if 'cuda' in path.lower():
            print(path)


if __name__ == "__main__":
    # Try to set CUDA path in multiple ways
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp"
    ]

    for path in cuda_paths:
        if os.path.exists(path):
            os.environ["CUDA_PATH"] = path
            if path.endswith('bin'):
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

    check_cuda_environment()