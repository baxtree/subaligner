########################
Advanced Usage
########################

You can train a new model with your own audiovisual files and in-sync subtitle files using Subaligner CLI. Thereafter,
the model can be imported and used for synchronising out-of-sync subtitles.

**Start fresh training**::

    $ subaligner_train -vd av_directory -sd subtitle_directory -tod training_output_directory

Make sure each subtitle file and its companion audiovisual file are sharing the same base filename, e.g.,
"awesome.mp4" and "awesome.srt" share the base filename "awesome". Then split them into two separate folders, e.g.,
av_directory and subtitle_directory, which are passed in with `-vd` or `--video_directory` and `-sd` or `--subtitle_directory`
, respectively. You need to also specify an output directory with `-tod` or `--training_output_directory` and it will hold
the results after training is finished and make sure it is writable to Subaligner, e.g., training_output_directory.

**Resume training**::

    $ subaligner_train -vd av_directory -sd subtitle_directory -tod training_output_directory -e 200 -r

Training over a large dataset is usually an expensive process and time consuming. You can stop the training and resume it with
`-r` or `--resume` at another convenient time to enhance an existing model stored in the aforementioned training output
directory. Note that the number of epochs you pass in with `-e` or `--epochs` needs to be greater than the number of epochs
already completed in the past. If the number is forgotten, you can pass in `-dde` or `--display_done_epochs` to recall it.

**Display completed epochs**::

    $ subaligner_train -dde -tod training_output_directory

Also note that on training resumption, av_directory and subtitle_directory will be ignored due to the reuse of feature
embedding by default.

**Reuse embeddings**::

    $ subaligner_train -utd -tod training_output_directory

Embeddings extracted from your media files can be reused with `-utd` or `--use_training_dump`. With that flag on, you can train a new
model of another kind (instead of re-using the same model on training resumption) without going through the feature embedding process,
which could take quite long to finish for a large dataset so as to be unnecessary if there is no change on it.

**Ignore sound effects**::

    $ subaligner_train -vd av_directory -sd subtitle_directory -tod training_output_directory --sound_effect_start_marker "(" --sound_effect_end_marker ")"

It is not uncommon that subtitles sometimes contain sound effects (e.g., "BARK", "(applause)" and "[MUSIC]", etc.). For limited training
data sets and not sophisticated enough network architectures, the model usually cannot capture all the sound effects very well.
To filter out sound effect subtitles and only preserve the vocal ones, you can pass in `-sesm` or `--sound_effect_start_marker` and/or
`seem` or `--sound_effect_end_marker` with strings which will be used by subaligner for finding sound effects and ignoring them within the training process.
For example, the above exemplary command will treat any strings starting with "(" and ending with ")" as sound effects.

**Train with embedded subtitles**::

    $ subaligner_train -vd av_directory -ess embedded:stream_index=0,file_extension=srt -tod training_output_directory

If your audiovisual files all contain embedded subtitles or teletexts of the same format and have been encoded in the same fashion, `-sd` or `--subtitle_directory`
can be omitted and subtitles will be extracted based on the specified subtitle selector. For instance, "embedded:stream_index=0,file_extension=srt"
can be passed in with `-ess` or `--embedded_subtitle_selector` and indicates that the embedded subtitle is located at
stream 0 (the first subtitle track) and will be extracted as a SubRip file pre and post synchronisation, while
"embedded:page_num=888,file_extension=srt" means the teletext is located on page 888. When `-sd` or `--subtitle_directory`
is present, make sure the folder passed in is empty.

**Run alignments after training**::

    $ subaligner -m single -v video.mp4 -s subtitle.srt -tod training_output_directory
    $ subaligner -m dual -v video.mp4 -s subtitle.srt -tod training_output_directory

To apply your trained model to subtitle alignment, pass in the training_output_directory containing training results as
shown above with `-tod` or `--training_output_directory`.

You can customise timeouts on media file processing and feature embedding to your needs using the following options:
**Re-configure timeouts**::

  -fet FEATURE_EMBEDDING_TIMEOUT, --feature_embedding_timeout FEATURE_EMBEDDING_TIMEOUT
                        Maximum waiting time in seconds when embedding features of media files
  -mpt MEDIA_PROCESS_TIMEOUT, --media_process_timeout MEDIA_PROCESS_TIMEOUT
                        Maximum waiting time in seconds when processing media files

**Hyperparameters**::

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

You can pass in the above flags to manually change hyperparameters for each training cycle. Alternatively, you can let
Subaligner tune hyperparameters automatically and the how-to is shown below.

**Hyperparameters tuning**::

     $ subaligner_tune -vd av_directory -sd subtitle_directory -tod training_output_directory

Subaligner has used the `Tree-structured Parzen Estimator Approach (TPE) <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ to
automatically run trails on different settings of hyper-parameter values and recommend the best one. You can pass in the following
flags to customise the configuration on tuning:

**Optional custom flags**::

    -ept EPOCHS_PER_TRAIL, --epochs_per_trail EPOCHS_PER_TRAIL
        Number of training epochs for each trial
    -t TRAILS, --trails TRAILS
        Number of tuning trials
    -nt {lstm,bi_lstm,conv_1d}, --network_type {lstm,bi_lstm,conv_1d}
        Network type
    -utd, --use_training_dump
        Use training dump instead of files in the video or subtitle directory

**Convert the subtitle to another format**::

    $ subaligner_convert -i subtitle.srt -o subtitle.vtt

**Convert the subtitle to another format and translate**::

    $ subaligner_convert --languages
    $ subaligner_convert -i subtitle_en.srt -o subtitle_zh.vtt -t eng,zho

**Translate the subtitle without changing the format**::

    $ subaligner_convert --languages
    $ subaligner_convert -i subtitle_en.srt -o subtitle_es.srt -t eng,spa

For output subtitles like MicroDVD relying on the frame rate, its value needs to be passed in with `-fr` or `--frame_rate`.

**On Windows with Docker Desktop**::

    docker run -v "/d/media":/media -w "/media" -it baxtree/subaligner COMMAND

The aforementioned commands can be run with `Docker Desktop <https://docs.docker.com/docker-for-windows/install/>`_ on Windows. Nonetheless, it is recommended to use Windows Subsystem for Linux (`WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_) to install Subaligner.