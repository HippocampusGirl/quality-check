import argparse
from pathlib import Path

from schema import Datastore

def parse_arguments():
    arg_parser = argparse.ArgumentParser(description="Arguments for image validation")
    arg_parser.add_argument("--database_path", required=True)
    arg_parser.add_argument("--output", default="quality_check_anomalies", required=False)
    arg_parser.add_argument("--method", choices=['intensities', 'average_image'], default='average_image', 
                            help="Choose the outlier detection method")
    arg_parser.add_argument("--threshold", type=float, default=3.0, 
                            help="Z-score threshold for outlier detection")

    return arg_parser.parse_args()

def main():
    args = parse_arguments()

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    datastore = Datastore(args.database_path)
    with datastore:
        all_outliers = datastore.detect_all_outliers(args.method, args.threshold)

        for suffix, outliers in all_outliers.items():
            print(f"Outliers for suffix '{suffix}':")
            for outlier in outliers:
                print(f"  Direction: {outlier[1]}, Index: {outlier[2]}, Image ID: {outlier[3]}")
                datastore.save_outlier_plot(outlier, output_folder)

if __name__ == "__main__":
    main()