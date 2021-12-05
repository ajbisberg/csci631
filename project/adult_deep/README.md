usage: main.py [-h] --data-path DATA_PATH --dataset {HOUSING,ADULT} [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE] [-lr LEARNING_RATE] [--use-dp] [--target-epsilon TARGET_EPSILON]
               [--target-delta TARGET_DELTA] [--max-grad-norm MAX_GRAD_NORM] [--save-model]

Train deep model

optional arguments:

  -h, --help                      show this help message and exit

  --data-path DATA_PATH           Path to pre-processed dataset

  --dataset {HOUSING,ADULT}       Choice of dataset, adult or housing

  --num-epochs NUM_EPOCHS         Number of epochs in training

  --batch-size BATCH_SIZE         Training batch size for SGD

  -lr LEARNING_RATE, --learning-rate LEARNING_RATE<br>
                                   Optimizer learning rate

  --use-dp                         Whether to use SGD or DP-SGD

  --target-epsilon TARGET_EPSILON  Target epsilon for DP-SGD

  --target-delta TARGET_DELTA      Target delta for DP-SGD

  --max-grad-norm MAX_GRAD_NORM    Max gradient norm for DP-SGD

  --save-model                           Write trained model to the disk

