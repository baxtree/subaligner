Feature: Subaligner CLI
    As a user of Subaligner
    I want to align my subtitle file to its companion video using the command line interface

    @video-input @with-mode
    Scenario Outline: Test alignments with modes
        Given I have a video file "test.mp4"
        And I have a subtitle file <subtitle-in>
        When I run the alignment with <aligner> on them with <mode> stage
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |  aligner          |  mode     |  subtitle-in      |  subtitle-out             |
        |  subaligner       |  single   |  "test.srt"       |  "test_aligned.srt"       |
        |  subaligner       |  single   |  "test.ttml"      |  "test_aligned.ttml"      |
        |  subaligner       |  single   |  "test.vtt"       |  "test_aligned.vtt"       |
        |  subaligner       |  single   |  "test.ssa"       |  "test_aligned.ssa"       |
        |  subaligner       |  single   |  "test.ass"       |  "test_aligned.ass"       |
        |  subaligner       |  single   |  "test.sub"       |  "test_aligned.sub"       |
        |  subaligner       |  single   |  "test.mpl2.txt"  |  "test.mpl2_aligned.txt"  |
        |  subaligner       |  single   |  "test.tmp"       |  "test_aligned.tmp"       |
        |  subaligner       |  dual     |  "test.srt"       |  "test_aligned.srt"       |
        |  subaligner       |  dual     |  "test.ttml"      |  "test_aligned.ttml"      |
        |  subaligner       |  dual     |  "test.vtt"       |  "test_aligned.vtt"       |
        |  subaligner       |  dual     |  "test.ssa"       |  "test_aligned.ssa"       |
        |  subaligner       |  dual     |  "test.ass"       |  "test_aligned.ass"       |
        |  subaligner       |  dual     |  "test.sub"       |  "test_aligned.sub"       |
        |  subaligner       |  dual     |  "test.mpl2.txt"  |  "test.mpl2_aligned.txt"  |
        |  subaligner       |  dual     |  "test.tmp"       |  "test_aligned.tmp"       |

    @video-input @without-mode
    Scenario Outline: Test alignments with video
        Given I have a video file "test.mp4"
        And I have a subtitle file <subtitle-in>
        When I run the alignment with <aligner> on them with <mode> stage
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |  aligner          |  mode     |  subtitle-in |  subtitle-out          |
        |  subaligner_1pass |  <NULL>   |  "test.srt"  |  "test_aligned.srt"    |
        |  subaligner_1pass |  <NULL>   |  "test.ttml" |  "test_aligned.ttml"   |
        |  subaligner_1pass |  <NULL>   |  "test.vtt"  |  "test_aligned.vtt"    |
        |  subaligner_1pass |  <NULL>   |  "test.ssa"  |  "test_aligned.ssa"    |
        |  subaligner_1pass |  <NULL>   |  "test.ass"  |  "test_aligned.ass"    |
        |  subaligner_2pass |  <NULL>   |  "test.srt"  |  "test_aligned.srt"    |
        |  subaligner_2pass |  <NULL>   |  "test.ttml" |  "test_aligned.ttml"   |
        |  subaligner_2pass |  <NULL>   |  "test.vtt"  |  "test_aligned.vtt"    |
        |  subaligner_2pass |  <NULL>   |  "test.ssa"  |  "test_aligned.ssa"    |
        |  subaligner_2pass |  <NULL>   |  "test.ass"  |  "test_aligned.ass"    |

    @audio-input
    Scenario Outline: Test alignments with audio
        Given I have a video file "test.wav"
        And I have a subtitle file "test.srt"
        When I run the alignment with <aligner> on them with <mode> stage
        Then a new subtitle file "test_aligned.srt" is generated
    Examples:
        |  aligner          |  mode     |
        |  subaligner_1pass |  <NULL>   |
        |  subaligner_2pass |  <NULL>   |
        |  subaligner       |  single   |
        |  subaligner       |  dual     |

    @custom-output
    Scenario Outline: Test alignments with custom output
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the alignment with <aligner> on them with <mode> stage and output "custom_aligned.srt"
        Then a new subtitle file "custom_aligned.srt" is generated
    Examples:
        |  aligner          |  mode     |
        |  subaligner_1pass |  <NULL>   |
        |  subaligner_2pass |  <NULL>   |
        |  subaligner       |  single   |
        |  subaligner       |  dual     |

    @no-stretch
    Scenario Outline: Test dual-stage alignment without stretch
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the alignment with <aligner> on them with <mode> stage and without stretch
        Then a new subtitle file "test_aligned.srt" is generated
    Examples:
        |  aligner          |  mode     |
        |  subaligner_2pass |  <NULL>   |
        |  subaligner       |  dual     |

    @quality-management
    Scenario Outline: Test exit when alignment log loss is too high
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        And I set the max log loss to "0.000001"
        When I run the alignment with <aligner> on them with <mode> stage and max loss
        Then it exits with code "22"
    Examples:
        |  aligner          |  mode     |
        |  subaligner_1pass |  <NULL>   |
        |  subaligner_2pass |  <NULL>   |
        |  subaligner       |  single   |
        |  subaligner       |  dual     |

    @strict
    Scenario Outline: Test dual-stage alignment with exiting on segment failures
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the alignment with <aligner> on them with <mode> stage and with exit_segfail
        Then a new subtitle file "test_aligned.srt" is generated
    Examples:
        |  aligner          |  mode     |
        |  subaligner_2pass |  <NULL>   |
        |  subaligner       |  dual     |

    @custom_model
    Scenario Outline: Test alignments with trained custom models
        Given I have a video file "test.wav"
        And I have a subtitle file "test.srt"
        When I run the alignment with <aligner> on them with <mode> stage and a custom model
        Then a new subtitle file "test_aligned.srt" is generated
    Examples:
        |  aligner          |  mode     |
        |  subaligner_1pass |  <NULL>   |
        |  subaligner_2pass |  <NULL>   |
        |  subaligner       |  single   |
        |  subaligner       |  dual     |

    @exception
    Scenario Outline: Test errors out on unsupported subtitle input
        Given I have a video file "test.mp4"
        And I have an unsupported subtitle file
        When I run the alignment with <aligner> on them with <mode> stage
        Then it exits with code "23"
    Examples:
        |  aligner          |  mode     |
        |  subaligner_1pass |  <NULL>   |
        |  subaligner_2pass |  <NULL>   |
        |  subaligner       |  single   |
        |  subaligner       |  dual     |

    @exception
    Scenario Outline: Test errors out on unsupported audiovisual input
        Given I have an unsupported video file
        And I have a subtitle file "test.srt"
        When I run the alignment with <aligner> on them with <mode> stage
        Then it exits with code "24"
    Examples:
        |  aligner          |  mode     |
        |  subaligner_1pass |  <NULL>   |
        |  subaligner_2pass |  <NULL>   |
        |  subaligner       |  single   |
        |  subaligner       |  dual     |

    @help
    Scenario Outline: Test help information display
        When I run the <aligner> command with help
        Then <aligner> help information is displayed
    Examples:
        |  aligner          |
        |  subaligner_1pass |
        |  subaligner_2pass |
        |  subaligner       |
