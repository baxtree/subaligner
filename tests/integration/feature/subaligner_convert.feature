Feature: Subaligner CLI
  As a user of Subaligner
  I want to convert subtitles from one format to another

  Scenario Outline: Test subtitle conversion
    Given I have a subtitle file <subtitle-in>
    When I run the converter with <subtitle-out> as the output
    Then a new subtitle file <subtitle-out> is generated
    Examples:
      |  subtitle-in      |  subtitle-out         |
      |  "test.srt"       |  "test_srt.ttml"      |
      |  "test.srt"       |  "test_srt.xml"       |
      |  "test.srt"       |  "test_srt.dfxp"      |
      |  "test.srt"       |  "test_srt.vtt"       |
      |  "test.srt"       |  "test_srt.ssa"       |
      |  "test.srt"       |  "test_srt.ass"       |
      |  "test.srt"       |  "test_mpl2_srt.txt"  |
      |  "test.srt"       |  "test_srt.tmp"       |
      |  "test.srt"       |  "test_srt.smi"       |
      |  "test.srt"       |  "test_srt.sami"      |
      |  "test.srt"       |  "test_srt.stl"       |
      |  "test.srt"       |  "test_srt.sub"       |
      |  "test.srt"       |  "test_srt.scc"       |
      |  "test.srt"       |  "test_srt.sbv"       |
      |  "test.srt"       |  "test_srt.ytt"       |
      |  "test.srt"       |  "test_srt.json"      |
      |  "test.ttml"      |  "test_ttml.srt"      |
      |  "test.xml"       |  "test_xml.srt"       |
      |  "test.dfxp"      |  "test_dfxp.srt"      |
      |  "test.vtt"       |  "test_vtt.srt"       |
      |  "test.ssa"       |  "test_ssa.srt"       |
      |  "test.ass"       |  "test_ass.srt"       |
      |  "test_mpl2.txt"  |  "test_mpl2_txt.srt"  |
      |  "test.tmp"       |  "test_tmp.srt"       |
      |  "test.smi"       |  "test_smi.srt"       |
      |  "test.sami"      |  "test_sami.srt"      |
      |  "test.stl"       |  "test_stl.srt"       |
      |  "test.sub"       |  "test_srt.srt"       |
      |  "test.scc"       |  "test_scc.xml"       |
      |  "test.sbv"       |  "test_sbv.srt"       |
      |  "test.ytt"       |  "test_ytt.srt"       |
      |  "test.json"      |  "test_ytt.srt"       |
      |  "test.ttml"      |  "test_ttml.vtt"      |
      |  "test.xml"       |  "test_xml.vtt"       |
      |  "test.dfxp"      |  "test_dfxp.vtt"      |
      |  "test.vtt"       |  "test_vtt.ttml"      |
      |  "test.ssa"       |  "test_ssa.ass"       |
      |  "test.ass"       |  "test_ass.ssa"       |
      |  "test_mpl2.txt"  |  "test_mpl2_txt.tmp"  |
      |  "test.tmp"       |  "test_tmp_mpl2.txt"  |
      |  "test.smi"       |  "test_smi.stl"       |
      |  "test.sami"      |  "test_sami.stl"      |
      |  "test.stl"       |  "test_stl.smi"       |
      |  "test.scc"       |  "test_scc.sub"       |

  Scenario: Test conversion from a remote subtitle file
    Given I have a subtitle file "https://raw.githubusercontent.com/baxtree/subaligner/master/tests/subaligner/resource/test.srt"
    When I run the converter with "test_srt.ttml" as the output
    Then a new subtitle file "test_srt.ttml" is generated

  Scenario Outline: Test subtitle conversion with translation
    Given I have a subtitle file <subtitle-in>
    When I run the converter with <language_pair> for translation and <subtitle-out> as the output
    Then a new subtitle file <subtitle-out> is generated
    Examples:
      |  subtitle-in      | language_pair | subtitle-out        |
      |  "test.srt"       | eng,zho       | "test_zh_srt.ttml"  |
      |  "test.srt"       | eng,spa       | "test_es_srt.ttml"  |
      |  "test.srt"       | eng,hin       | "test_hi_srt.ttml"  |
      |  "test.srt"       | eng,fra       | "test_fr_srt.ttml"  |
      |  "test.srt"       | eng,ara       | "test_ar_srt.ttml"  |
      |  "test.srt"       | eng,jpn       | "test_ja_srt.ttml"  |
      |  "test.srt"       | eng,rus       | "test_ru_srt.ttml"  |
      |  "test.srt"       | eng,ind       | "test_id_srt.ttml"  |

  Scenario: Test language codes display
      When I run the subaligner_convert command with languages
      Then supported language codes are displayed
