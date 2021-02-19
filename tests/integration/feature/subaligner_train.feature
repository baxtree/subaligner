Feature: Subaligner CLI
    As an advanced user of Subaligner
    I want to train a new model with my own audiovisual and subtitle files and tune hyperparameters

    @train @lstm
    Scenario: Test training on the LSTM network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_train against them with the following options
            """
            -bs 10 -do 0.5 -e 2 -p 1 -fhs 10 -bhs 5,2 -lr 0.01 -nt lstm -vs 0.3 -o adam
            """
        Then a model and a training log file are generated
        When I run the subaligner_train to display the finished epochs
        Then it shows the done epochs equal to 2

    @train @bi-lstm
    Scenario: Test training on the Bidirectional LSTM network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_train against them with the following options
            """
            -bs 10 -do 0.5 -e 3 -p 1 -fhs 10 -bhs 5,2 -lr 0.01 -nt bi_lstm -vs 0.3 -o adam
            """
        Then a model and a training log file are generated
        When I run the subaligner_train to display the finished epochs
        Then it shows the done epochs equal to 3

    @train @conv-1d
    Scenario: Test training on the Conv1D network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_train against them with the following options
            """
            -bs 10 -do 0.5 -e 2 -p 1 -fhs 10 -bhs 5,2 -lr 0.01 -nt conv_1d -vs 0.3 -o adam
            """
        Then a model and a training log file are generated
        When I run the subaligner_train to display the finished epochs
        Then it shows the done epochs equal to 2

    @train @ignore-sound-effects
    Scenario: Test ignoring sound effects during on training
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_train against them with the following options
            """
            -e 2 -nt lstm --sound_effect_start_marker "(" --sound_effect_end_marker ")"
            """
        Then a model and a training log file are generated
        When I run the subaligner_train to display the finished epochs
        Then it shows the done epochs equal to 2

    @train @ignore-sound-effects
    Scenario: Test erroring on sound_effect_end_marker used alone
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_train against them with the following options
            """
            -e 2 -nt lstm --sound_effect_end_marker ")"
            """
        Then it exits with code "21"

    @hyperparameter-tuning @lstm
    Scenario: Test hyperparameter tuning on the LSTM network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_tune against them with the following flags
            | epoch_per_trail   | trails    | network_type  |
            | 1                 | 2         | lstm          |
        Then a hyperparameter file is generated

    @hyperparameter-tuning @bi-lstm
    Scenario: Test hyperparameter tuning on the Bidirectional LSTM network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_tune against them with the following flags
            | epoch_per_trail   | trails    | network_type  |
            | 2                 | 1         | bi_lstm       |
        Then a hyperparameter file is generated

    @hyperparameter-tuning @conv-1d
    Scenario: Test hyperparameter tuning on the Conv1D network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_tune against them with the following flags
            | epoch_per_trail   | trails    | network_type  |
            | 1                 | 2         | conv_1d       |
        Then a hyperparameter file is generated
