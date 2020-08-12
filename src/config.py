MODEL_NAME = 'distilbert-base-uncased' # pretrained model from Transformers
LOG_DIR = "./logdir_amazon"    # for training logs and tensorboard visualizations
NUM_EPOCHS = 1                         # smth around 2-6 epochs is typically fine when finetuning transformers
BATCH_SIZE = 32                       # depends on your available GPU memory (in combination with max seq length)
MAX_SEQ_LENGTH = 512                   # depends on your available GPU memory (in combination with batch size)
LEARN_RATE = 3e-5                      # learning rate is typically ~1e-5 for transformers
ACCUM_STEPS = 3                        # one optimization step for that many backward passes
SEED = 42                              # random seed for reproducibility
fp16_params = dict(opt_level="O1")