Feature: Subaligner CLI
    As an advanced user of Subaligner
    I want to train a new model with my own audiovisual and subtitle files

    @train
    Scenario: Test alignments with modes
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the trained model in directory "output"
        When I run the subaligner_train against them with the following hyper parameters
            """
            -bs 10 -do 0.5 -e 2 -p 1 -fhs 10 -bhs 5,2 -lr 0.01 -nt lstm -vs 0.3 -o adagrad
            """
        Then a model and a training log file are generated