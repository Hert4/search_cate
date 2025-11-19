import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config.settings import DEFAULT_DATA_PATH, DEVICE
from evaluation.compare_scalable_methods import compare_scalable_methods


def main():
    """
    Compare all 7 methods including the scalable ones
    """
    print("Running comparison of all 7 methods...")
    print("This includes both traditional and scalable methods for 100K+ categories")

    # Run comparison
    results = compare_scalable_methods(
        csv_path=DEFAULT_DATA_PATH,
        device=DEVICE,
        verbose=True
    )


if __name__ == "__main__":
    main()