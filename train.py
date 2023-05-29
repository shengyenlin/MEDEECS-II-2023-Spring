import tensorflow as tf
import keras
import argparse
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from src.metrics import print_metric
import time

def parse_args():
    parser.add_argument(
        '--weight_path', type=str, 
        )

    parser = argparse.ArgumentParser()

def main():
    args = parse_args()
    model = load_model(args.weight_path)

    # load df

    # build 

    print(model)

if __name__ == "__main__":
    main()