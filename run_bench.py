import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config.settings import DEFAULT_DATA_PATH, DEVICE
from evaluation.compare_methods import compare_adaptive_methods


def main():

    # Run comparison
    results = compare_adaptive_methods(
        csv_path=DEFAULT_DATA_PATH,
        device=DEVICE,
        verbose=True 
    )



if __name__ == "__main__":
    main()