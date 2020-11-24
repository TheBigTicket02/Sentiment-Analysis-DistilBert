import torch.nn as nn
from torch.utils.data import DataLoader
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, OptimizerCallback, CheckpointCallback
from catalyst.contrib.nn import RAdam, Lookahead, OneCycleLRWithWarmup
import model
import config
import dataset

def main():
    train_dataset = dataset.SentimentDataset(
        texts=df_train['sentences'].values.tolist(),
        labels=df_train['labels'].values,
        max_seq_length=config.MAX_SEQ_LENGTH,
        model_name=config.MODEL_NAME
    )

    valid_dataset = dataset.SentimentDataset(
        texts=df_valid['sentences'].values.tolist(),
        labels=df_valid['labels'].values,
        max_seq_length=config.MAX_SEQ_LENGTH,
        model_name=config.MODEL_NAME
    )

    train_val_loaders = {
        "train": DataLoader(dataset=train_dataset,
                            batch_size=config.BATCH_SIZE, 
                            shuffle=True, num_workers=2, pin_memory=True),
        "valid": DataLoader(dataset=valid_dataset,
                            batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=2, pin_memory=True)    
    }

    dBert = model.DistilBert()

    param_optim = list(dBert.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    criterion = nn.CrossEntropyLoss()

    base_optimizer = RAdam([
        {'params': [p for n,p in param_optim if not any(nd in n for nd in no_decay)],
        'weight_decay': config.WEIGHT_DECAY}, 
        {'params': [p for n,p in param_optim if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0}
    ])
    optimizer = Lookahead(base_optimizer)
    scheduler = OneCycleLRWithWarmup(
        optimizer, 
        num_steps=config.NUM_EPOCHS, 
        lr_range=(config.LEARNING_RATE, 1e-8),
        init_lr=config.LEARNING_RATE,
        warmup_steps=0,
    )
    runner = SupervisedRunner(
        input_key=(
            "input_ids",
            "attention_mask"
        )
    )
    # model training
    runner.train(
        model=dBert,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=train_val_loaders,
        callbacks=[
            AccuracyCallback(num_classes=2),
            OptimizerCallback(accumulation_steps=config.ACCUM_STEPS),
        ],
        fp16=config.FP_16,
        logdir=config.LOG_DIR,
        num_epochs=config.NUM_EPOCHS,
        verbose=True
    )

if __name__ == '__main__':
    main()