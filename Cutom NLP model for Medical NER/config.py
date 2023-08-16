import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 20
BASE_MODEL_PATH = "dmis-lab/biobert-base-cased-v1.1"
MODEL_PATH = "model.bin"
TRAINING_FILE = "data/NCBI_corpus_training.txt"
TESTING_FILE ="data/NCBI_corpus_testing.txt"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)