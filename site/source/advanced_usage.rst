########################
Advanced Usage
########################

You can train a new model with your own audiovisual files and subtitle files and the model can be
later used for synchronising subtitles.

**Start a fresh Training**::

    (.venv) $ subaligner_train -vd av_directory -sd subtitle_directory -od output_directory

Make sure each subtitle and its companion audiovisual content are sharing the same base filename, e.g.,
awesome.mp4 and awesome.srt. Than spit them into a two saperate folders, e.g., av_directory and subtitle_directory.
The training result will be stored into a specified directory, e.g., output_directory.

**Resume training**::

    (.venv) $ subaligner_train -vd av_directory -sd subtitle_directory -od output_directory -e 200 -r

Training over a large dataset is normally an expensive process. You can resume the previous training with `-r` or `--resume`.
Make sure the number of epochs you pass in is greater than the epochs done in the past.

**Reuse embeddings**::

    (.venv) $ subaligner_train -vd av_directory -sd subtitle_directory -od output_directory -utd

Embeddings extracted from your media files can be reused with `-utd` or `--use_training_dump`. With the flag on, you can train a new
model of another kind without going through feature embedding process, which could take quite long to finish.

**Hyper parameters**::

    -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of 32ms samples at each training step
    -do DROPOUT, --dropout DROPOUT
        Dropout rate between 0 and 1 used at the end of each intermediate hidden layer
    -e EPOCHS, --epochs EPOCHS
        Total training epochs
    -p PATIENCE, --patience PATIENCE
        Number of epochs with no improvement after which training will be stopped
    -fhs FRONT_HIDDEN_SIZE, --front_hidden_size FRONT_HIDDEN_SIZE
        Number of neurons in the front LSTM or Conv1D layer
    -bhs BACK_HIDDEN_SIZE, --back_hidden_size BACK_HIDDEN_SIZE
        Comma-separated numbers of neurons in the back Dense layers
    -lr LEARNING_RATE, --learning_rate LEARNING_RATE
        Learning rate of the optimiser
    -nt {lstm,bi_lstm,conv_1d}, --network_type {lstm,bi_lstm,conv_1d}
        Network type
    -vs VALIDATION_SPLIT, --validation_split VALIDATION_SPLIT
        Fraction between 0 and 1 of the training data to be used as validation data
    -o {adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd}, --optimizer {adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd}
        TensorFlow optimizer

You can pass in the above flags to manually change hyper parameters before each training. Or You can let Subaligner tune hyper parameters automatically as shown below.

**Hyper parameters tuning**::

     (.venv) $ subaligner_tune -vd av_directory -sd subtitle_directory -od output_directory

You can pass in the following flags to customise the tuning settings:

**Optional custom flags**::

    -ept EPOCHS_PER_TRAIL, --epochs_per_trail EPOCHS_PER_TRAIL
        Number of training epochs for each trial
    -t TRAILS, --trails TRAILS
        Number of tuning trials
    -nt {lstm,bi_lstm,conv_1d}, --network_type {lstm,bi_lstm,conv_1d}
        Network type
    -utd, --use_training_dump
        Use training dump instead of files in the video or subtitle directory

