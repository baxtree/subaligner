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
        |  subaligner       |  single   |  "test.xml"       |  "test_aligned.xml"       |
        |  subaligner       |  single   |  "test.dfxp"      |  "test_aligned.dfxp"      |
        |  subaligner       |  single   |  "test.vtt"       |  "test_aligned.vtt"       |
        |  subaligner       |  single   |  "test.ssa"       |  "test_aligned.ssa"       |
        |  subaligner       |  single   |  "test.ass"       |  "test_aligned.ass"       |
        |  subaligner       |  single   |  "test.sub"       |  "test_aligned.sub"       |
        |  subaligner       |  single   |  "test_mpl2.txt"  |  "test_mpl2_aligned.txt"  |
        |  subaligner       |  single   |  "test.tmp"       |  "test_aligned.tmp"       |
        |  subaligner       |  single   |  "test.smi"       |  "test_aligned.smi"       |
        |  subaligner       |  single   |  "test.sami"      |  "test_aligned.sami"      |
        |  subaligner       |  single   |  "test.stl"       |  "test_aligned.srt"       |
        |  subaligner       |  single   |  "test.scc"       |  "test_aligned.srt"       |
        |  subaligner       |  single   |  "test.sbv"       |  "test_aligned.sbv"       |
        |  subaligner       |  single   |  "test.ytt"       |  "test_aligned.ytt"       |
        |  subaligner       |  single   |  "test.json"      |  "test_aligned.json"      |
        |  subaligner       |  single   |  "accented.srt"   |  "accented_aligned.srt"   |
        |  subaligner       |  dual     |  "test.srt"       |  "test_aligned.srt"       |
        |  subaligner       |  dual     |  "test.ttml"      |  "test_aligned.ttml"      |
        |  subaligner       |  dual     |  "test.xml"       |  "test_aligned.xml"       |
        |  subaligner       |  dual     |  "test.dfxp"      |  "test_aligned.dfxp"      |
        |  subaligner       |  dual     |  "test.vtt"       |  "test_aligned.vtt"       |
        |  subaligner       |  dual     |  "test.ssa"       |  "test_aligned.ssa"       |
        |  subaligner       |  dual     |  "test.ass"       |  "test_aligned.ass"       |
        |  subaligner       |  dual     |  "test.sub"       |  "test_aligned.sub"       |
        |  subaligner       |  dual     |  "test_mpl2.txt"  |  "test_mpl2_aligned.txt"  |
        |  subaligner       |  dual     |  "test.tmp"       |  "test_aligned.tmp"       |
        |  subaligner       |  dual     |  "test.smi"       |  "test_aligned.smi"       |
        |  subaligner       |  dual     |  "test.sami"      |  "test_aligned.sami"      |
        |  subaligner       |  dual     |  "test.stl"       |  "test_aligned.srt"       |
        |  subaligner       |  dual     |  "test.scc"       |  "test_aligned.scc"       |
        |  subaligner       |  dual     |  "test.sbv"       |  "test_aligned.sbv"       |
        |  subaligner       |  dual     |  "test.ytt"       |  "test_aligned.ytt"       |
        |  subaligner       |  dual     |  "test.json"      |  "test_aligned.json"       |
        |  subaligner       |  dual     |  "accented.srt"   |  "accented_aligned.srt"   |

    @video-input @without-mode
    Scenario Outline: Test alignments without modes
        Given I have a video file "test.mp4"
        And I have a subtitle file <subtitle-in>
        When I run the alignment with <aligner> on them with <mode> stage
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |  aligner          |  mode     |  subtitle-in      |  subtitle-out             |
        |  subaligner_1pass |  <NULL>   |  "test.srt"       |  "test_aligned.srt"       |
        |  subaligner_1pass |  <NULL>   |  "test.ttml"      |  "test_aligned.ttml"      |
        |  subaligner_1pass |  <NULL>   |  "test.xml"       |  "test_aligned.xml"       |
        |  subaligner_1pass |  <NULL>   |  "test.dfxp"      |  "test_aligned.dfxp"      |
        |  subaligner_1pass |  <NULL>   |  "test.vtt"       |  "test_aligned.vtt"       |
        |  subaligner_1pass |  <NULL>   |  "test.ssa"       |  "test_aligned.ssa"       |
        |  subaligner_1pass |  <NULL>   |  "test.ass"       |  "test_aligned.ass"       |
        |  subaligner_1pass |  <NULL>   |  "test.sub"       |  "test_aligned.sub"       |
        |  subaligner_1pass |  <NULL>   |  "test_mpl2.txt"  |  "test_mpl2_aligned.txt"  |
        |  subaligner_1pass |  <NULL>   |  "test.tmp"       |  "test_aligned.tmp"       |
        |  subaligner_1pass |  <NULL>   |  "test.smi"       |  "test_aligned.smi"       |
        |  subaligner_1pass |  <NULL>   |  "test.sami"      |  "test_aligned.sami"      |
        |  subaligner_1pass |  <NULL>   |  "test.stl"       |  "test_aligned.srt"       |
        |  subaligner_1pass |  <NULL>   |  "test.scc"       |  "test_aligned.scc"       |
        |  subaligner_1pass |  <NULL>   |  "test.sbv"       |  "test_aligned.sbv"       |
        |  subaligner_1pass |  <NULL>   |  "test.ytt"       |  "test_aligned.ytt"       |
        |  subaligner_2pass |  <NULL>   |  "test.srt"       |  "test_aligned.srt"       |
        |  subaligner_2pass |  <NULL>   |  "test.ttml"      |  "test_aligned.ttml"      |
        |  subaligner_1pass |  <NULL>   |  "test.xml"       |  "test_aligned.xml"       |
        |  subaligner_1pass |  <NULL>   |  "test.dfxp"      |  "test_aligned.dfxp"      |
        |  subaligner_2pass |  <NULL>   |  "test.vtt"       |  "test_aligned.vtt"       |
        |  subaligner_2pass |  <NULL>   |  "test.ssa"       |  "test_aligned.ssa"       |
        |  subaligner_2pass |  <NULL>   |  "test.ass"       |  "test_aligned.ass"       |
        |  subaligner_2pass |  <NULL>   |  "test.sub"       |  "test_aligned.sub"       |
        |  subaligner_2pass |  <NULL>   |  "test_mpl2.txt"  |  "test_mpl2_aligned.txt"  |
        |  subaligner_2pass |  <NULL>   |  "test.tmp"       |  "test_aligned.tmp"       |
        |  subaligner_2pass |  <NULL>   |  "test.smi"       |  "test_aligned.smi"       |
        |  subaligner_2pass |  <NULL>   |  "test.sami"      |  "test_aligned.sami"      |
        |  subaligner_2pass |  <NULL>   |  "test.stl"       |  "test_aligned.srt"       |
        |  subaligner_2pass |  <NULL>   |  "test.scc"       |  "test_aligned.scc"       |
        |  subaligner_2pass |  <NULL>   |  "test.sbv"       |  "test_aligned.sbv"       |
        |  subaligner_2pass |  <NULL>   |  "test.ytt"       |  "test_aligned.ytt"       |

    @with-mode @script
    Scenario Outline: Test alignments with plain texts as input
        Given I have a video file <video-in>
        And I have a subtitle file "test_plain.txt"
        When I run the alignment with subaligner on them with script stage and output <subtitle-out>
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |   video-in    |  subtitle-out             |
        |   "test.mp4"  |  "test_aligned.srt"       |
        |   "test.mp4"  |  "test_aligned.ttml"      |
        |   "test.mp4"  |  "test_aligned.xml"       |
        |   "test.mp4"  |  "test_aligned.dfxp"      |
        |   "test.mp4"  |  "test_aligned.vtt"       |
        |   "test.mp4"  |  "test_aligned.ssa"       |
        |   "test.mp4"  |  "test_aligned.ass"       |
        |   "test.mp4"  |  "test_aligned.sub"       |
        |   "test.wav"  |  "test_mpl2_aligned.txt"  |
        |   "test.wav"  |  "test_aligned.tmp"       |
        |   "test.wav"  |  "test_aligned.smi"       |
        |   "test.wav"  |  "test_aligned.sami"      |
        |   "test.wav"  |  "test_aligned.scc"       |
        |   "test.wav"  |  "test_aligned.sbv"       |
        |   "test.wav"  |  "test_aligned.ytt"       |
        |   "test.wav"  |  "test_aligned.json"      |

    @with-mode @script @timed-words
    Scenario Outline: Test alignments with plain texts as input and timed words as output
        Given I have a video file <video-in>
        And I have a subtitle file "test_plain.txt"
        When I run the alignment with subaligner on them with timed words and output <subtitle-out>
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |   video-in    |  subtitle-out             |
        |   "test.mp4"  |  "test_aligned.json"      |

    @remote-inputs
    Scenario Outline: Test alignments with remote inputs
        Given I have a video file "https://raw.githubusercontent.com/baxtree/subaligner/master/tests/subaligner/resource/test.mp4"
        And I have a subtitle file "https://raw.githubusercontent.com/baxtree/subaligner/master/tests/subaligner/resource/test.srt"
        When I run the alignment with <aligner> on them with <mode> stage and output "custom_aligned.srt"
        Then a new subtitle file "custom_aligned.srt" is generated
    Examples:
        |  aligner          |  mode     |
        |  subaligner_1pass |  <NULL>   |
        |  subaligner_2pass |  <NULL>   |
        |  subaligner       |  single   |
        |  subaligner       |  dual     |

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

    @embedded-subtitle
    Scenario Outline: Test alignments with embedded subtitles
        Given I have a video file <video-in>
        And I have selector <selector> for the embedded subtitle
        When I run the alignment with <aligner> on them with <mode> stage and output <subtitle-out>
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |  aligner          |  mode     |  selector                     |  subtitle-out         |   video-in    |
        |  subaligner_1pass |  <NULL>   |  "embedded:stream_index=0"    |  "test_aligned.srt"   |   "test.mkv"  |
        |  subaligner_2pass |  <NULL>   |  "embedded:stream_index=0"    |  "test_aligned.srt"   |   "test.mkv"  |
        |  subaligner       |  single   |  "embedded:stream_index=0"    |  "test_aligned.srt"   |   "test.mkv"  |
        |  subaligner       |  dual     |  "embedded:stream_index=0"    |  "test_aligned.srt"   |   "test.mkv"  |
#        |  subaligner_1pass |  <NULL>   |  "embedded:page_num=888"      |  "test_aligned.srt"   |   "test.ts"   |
#        |  subaligner_2pass |  <NULL>   |  "embedded:page_num=888"      |  "test_aligned.srt"   |   "test.ts"   |
#        |  subaligner       |  single   |  "embedded:page_num=888"      |  "test_aligned.srt"   |   "test.ts"   |
#        |  subaligner       |  dual     |  "embedded:page_num=888"      |  "test_aligned.srt"   |   "test.ts"   |

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

    @stretch
    Scenario Outline: Test dual-stage alignment with stretch on
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the alignment with <aligner> on them with <mode> stage and with stretch on
        Then a new subtitle file "test_aligned.srt" is generated
    Examples:
        |  aligner          |  mode     |
        |  subaligner_2pass |  <NULL>   |
        |  subaligner       |  dual     |

    @quality-control
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

    Scenario: Test alignments with the file path containing whitespace ([] == " ")
        Given I have a video file "test[]spaced.mp4"
        And I have a subtitle file "test[]spaced.vtt"
        When I run the alignment with subaligner on them with dual stage
        Then a new subtitle file "test[]spaced_aligned.vtt" is generated

    @translation
    Scenario Outline: Test translation on aligned subtitles
        Given I have a video file "test.mp4"
        And I have a subtitle file <subtitle-in>
        When I run the alignment with <aligner> on them with <mode> stage and <language-pair> for translation
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |  aligner          |  mode     |  subtitle-in      |   language-pair   |   subtitle-out            |
        |  subaligner       |  single   |  "test.srt"       |   eng,zho         |   "test_aligned.srt"      |
        |  subaligner       |  dual     |  "test.srt"       |   eng,spa         |   "test_aligned.srt"      |
        |  subaligner       |  script   |  "test_plain.txt" |   eng,ita         |   "test_aligned.srt"      |
        |  subaligner       |  script   |  "test_plain.txt" |   eng,por         |   "test_aligned.srt"      |
        |  subaligner_1pass |  <NULL>   |  "test.srt"       |   eng,fra         |   "test_aligned.srt"      |
        |  subaligner_2pass |  <NULL>   |  "test.srt"       |   eng,deu         |   "test_aligned.srt"      |

    @transcription @custom-prompt
    Scenario Outline: Test transcription on audiovisual input and subtitle generation
        Given I have a video file <video-in>
        When I run the alignment with <aligner> on them with <mode> stage with <language> language, <recipe> recipe and <flavour> flavour and prompt <prompt>
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |   video-in    |   aligner     |  mode         |   language    |   recipe      |   flavour     |   prompt      |   subtitle-out        |
        |   "test.mp4"  |   subaligner  |  transcribe   |   eng         |   whisper     |   tiny        |   <NULL>      |   "test_aligned.srt"  |
        |   "test.wav"  |   subaligner  |  transcribe   |   eng         |   whisper     |   tiny        |   test_prompt |   "test_aligned.srt"  |

    @transcription @subtitle-prompt
    Scenario Outline: Test transcription on audiovisual input with original subtitle as prompts and subtitle generation
        Given I have a video file <video-in>
        And I have a subtitle file <subtitle-in>
        When I run the alignment with <aligner> on them with <mode> stage with <language> language, <recipe> recipe and <flavour> flavour
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |   video-in    |   aligner     |  mode         |  subtitle-in      |   language    |   recipe      |   flavour |   subtitle-out        |
        |   "test.mp4"  |   subaligner  |  transcribe   |  "test.srt"       |   eng         |   whisper     |   tiny    |   "test_aligned.srt"  |

    @transcription @timed-words
    Scenario Outline: Test transcription on audiovisual input and subtitle generation with timed words
        Given I have a video file <video-in>
        And I have a subtitle file <subtitle-in>
        When I run the alignment with subaligner on them with transcribe stage and timed words
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |   video-in    |   subtitle-in |   subtitle-out        |
        |   "test.mp4"  |   "test.srt"  |   "test_aligned.json" |

    @batch
    Scenario Outline: Test batch alignment
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the alignment output in directory <output_dir>
        When I run the subaligner_batch on them with <mode> stage
        Then a new subtitle file <subtitle-out> is generated in the above output directory
    Examples:
        |  output_dir       |  mode     |  subtitle-out |
        |  "aligned_single" |  single   |  "test.srt"   |
        |  "aligned_dual"   |  dual     |  "test.srt"   |

    @batch
    Scenario Outline: Test batch alignment with output formats
        Given I have an audiovisual file directory "av"
        And I have a subtitle file directory "sub"
        And I want to save the alignment output in directory <output_dir> with format <subtitle-format>
        When I run the subaligner_batch on them with <mode> stage
        Then a new subtitle file <subtitle-out> is generated in the above output directory
    Examples:
        |  output_dir       |  mode     |   subtitle-format |  subtitle-out |
        |  "aligned_single" |  single   |   "ttml"          |  "test.ttml"  |
        |  "aligned_dual"   |  dual     |   "vtt"           |  "test.vtt"   |

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

    @exception @timeout
    Scenario Outline: Test timeout on processing media files
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the alignment with <aligner> on them with <mode> stage and a short timeout
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

    @languages
    Scenario Outline: Test language codes display
        When I run the <aligner> command with languages
        Then supported language codes are displayed
    Examples:
        |  aligner          |
        |  subaligner_1pass |
        |  subaligner_2pass |
        |  subaligner       |

    @manual_shift
    Scenario Outline: Shift the subtitle by offset in seconds
        Given I have a subtitle file <subtitle-in>
        When I run the manual shift with offset of <offset> in seconds
        Then a new subtitle file <subtitle-out> is generated
    Examples:
        |   subtitle-in |   subtitle-out             |  offset  |
        |   "test.srt"  |   "test_shifted.srt"       |  1.1     |
        |   "test.ttml" |   "test_shifted.ttml"      |  2.2     |
        |   "test.xml"  |   "test_shifted.xml"       |  3       |
        |   "test.dfxp" |   "test_shifted.dfxp"      |  4.25    |
        |   "test.vtt"  |   "test_shifted.vtt"       |  +0      |
        |   "test.sami" |   "test_shifted.sami"      |  0       |
        |   "test.ssa"  |   "test_shifted.ssa"       |  -0      |
        |   "test.ass"  |   "test_shifted.ass"       |  -1.1    |
        |   "test.sub"  |   "test_shifted.sub"       |  -2.2    |
        |   "test.tmp"  |   "test_shifted.tmp"       |  -3      |
        |   "test.smi"  |   "test_shifted.smi"       |  -4.25   |
        |   "test.scc"  |   "test_shifted.scc"       |  1.1     |
        |   "test.sbv"  |   "test_shifted.sbv"       |  2.2     |
        |   "test.ytt"  |   "test_shifted.ytt"       |  3       |
        |   "test.json" |   "test_shifted.json"      |  3       |
