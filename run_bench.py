import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config.settings import DEFAULT_DATA_PATH, DEVICE
from evaluation.compare_scalable_methods import compare_scalable_methods_only


def main():

    # Run comparison
    results = compare_scalable_methods_only(
        csv_path=DEFAULT_DATA_PATH,
        device=DEVICE,
        verbose=True
    )



if __name__ == "__main__":
    main()