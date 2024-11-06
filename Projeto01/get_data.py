import os


def main() -> None:
    if not os.path.exists("data"):
        os.makedirs("data")

    os.system(
        "kaggle datasets download balabaskar/wonders-of-the-world-image-classification -p data/ && kaggle datasets download paultimothymooney/chest-xray-pneumonia -p data/"
    )
    os.system(
        "unzip data/wonders-of-the-world-image-classification.zip -d data/ && unzip data/chest-xray-pneumonia.zip -d data/"
    )


if __name__ == "__main__":
    main()
