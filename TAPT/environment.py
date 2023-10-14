from random_search import RandomSearch
import os


VAMPIRE = {
        "LAZY_DATASET_READER": os.environ.get("LAZY", 0),
        "KL_ANNEALING": "linear",
        "KLD_CLAMP": None,
        "SIGMOID_WEIGHT_1": 0.25,
        "SIGMOID_WEIGHT_2": 15,
        "LINEAR_SCALING": 1000,
        "VAE_HIDDEN_DIM": 81,
        "TRAIN_PATH": os.environ["DATA_DIR"] + "/train.npz",
        "DEV_PATH": os.environ["DATA_DIR"] + "/dev.npz",
        "REFERENCE_COUNTS": os.environ["DATA_DIR"] + "/reference/ref.npz",
        "REFERENCE_VOCAB": os.environ["DATA_DIR"] + "/reference/ref.vocab.json",
        "VOCABULARY_DIRECTORY": os.environ["DATA_DIR"] + "/vocabulary/",
        "BACKGROUND_DATA_PATH": os.environ["DATA_DIR"] + "/vampire.bgfreq",
        "NUM_ENCODER_LAYERS": 2,
        "ENCODER_ACTIVATION": "relu",
        "MEAN_PROJECTION_ACTIVATION": "linear",
        "NUM_MEAN_PROJECTION_LAYERS": 1,
        "LOG_VAR_PROJECTION_ACTIVATION": "linear",
        "NUM_LOG_VAR_PROJECTION_LAYERS": 1,
        "SEED": RandomSearch.random_integer(0, 100000),
        "Z_DROPOUT": 0.49,
        "LEARNING_RATE": 1e-3,
        "TRACK_NPMI": True,
        "CUDA_DEVICE": 0,
        "UPDATE_BACKGROUND_FREQUENCY": 0,
        "VOCAB_SIZE": os.environ.get("VOCAB_SIZE", 30000),
        "BATCH_SIZE": 128,
        "MIN_SEQUENCE_LENGTH": 3,
        "NUM_EPOCHS": 30,
        "PATIENCE": 5,
        "VALIDATION_METRIC": "-loss"
}



ENVIRONMENTS = {
        'VAMPIRE': VAMPIRE,
}






