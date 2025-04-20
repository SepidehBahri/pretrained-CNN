import argparse
from src import train, evaluate

def main():
    parser = argparse.ArgumentParser(description="Pretrained CNN Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate"],
        required=True,
        help="Run mode: 'train' or 'evaluate'"
    )

    args = parser.parse_args()

    if args.mode == "train":
        print("starting training...")
        train.main()
    elif args.mode == "evaluate":
        print("starting evaluation...")
        evaluate.main()

if __name__ == "__main__":
    main()
