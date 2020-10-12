Feature: Subaligner CLI
    As an advanced user of Subaligner
    I want to train a new model with my own audiovisual and subtitle files and tune hyper parameters

    @train @lstm
    Scenario: Test training on the LSTM network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_train against them with the following hyper parameters
            """
            -bs 10 -do 0.5 -e 2 -p 1 -fhs 10 -bhs 5,2 -lr 0.01 -nt lstm -vs 0.3 -o adam
            """
        Then a model and a training log file are generated

    @train @bi-lstm
    Scenario: Test training on the Bidirectional LSTM network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_train against them with the following hyper parameters
            """
            -bs 10 -do 0.5 -e 2 -p 1 -fhs 10 -bhs 5,2 -lr 0.01 -nt bi_lstm -vs 0.3 -o adam
            """
        Then a model and a training log file are generated

    @train @conv-1d
    Scenario: Test training on the Conv1D network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_train against them with the following hyper parameters
            """
            -bs 10 -do 0.5 -e 2 -p 1 -fhs 10 -bhs 5,2 -lr 0.01 -nt conv_1d -vs 0.3 -o adam
            """
        Then a model and a training log file are generated

    @hyper-parameter-tuning @lstm
    Scenario: Test hyper parameter tuning on the LSTM network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_tune against them with the following flags
            | epoch_per_trail   | trails    | network_type  |
            | 1                 | 2         | lstm          |
        Then a hyper parameter file is generated

    @hyper-parameter-tuning @bi-lstm
    Scenario: Test hyper parameter tuning on the Bidirectional LSTM network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_tune against them with the following flags
            | epoch_per_trail   | trails    | network_type  |
            | 2                 | 1         | bi_lstm       |
        Then a hyper parameter file is generated

    @hyper-parameter-tuning @conv-1d
    Scenario: Test hyper parameter tuning on the Conv1D network
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the output in directory "output"
        When I run the subaligner_tune against them with the following flags
            | epoch_per_trail   | trails    | network_type  |
            | 1                 | 2         | conv_1d       |
        Then a hyper parameter file is generated
