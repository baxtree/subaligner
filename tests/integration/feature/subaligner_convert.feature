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
      |  "test.srt"       |  "test.mpl2_srt.txt"  |
      |  "test.srt"       |  "test_srt.tmp"       |
      |  "test.srt"       |  "test_srt.smi"       |
      |  "test.srt"       |  "test_srt.sami"      |
      |  "test.srt"       |  "test_srt.stl"       |
      |  "test.srt"       |  "test_srt.sub"       |
      |  "test.srt"       |  "test_srt.scc"       |
      |  "test.ttml"      |  "test_ttml.srt"      |
      |  "test.xml"       |  "test_xml.srt"       |
      |  "test.dfxp"      |  "test_dfxp.srt"      |
      |  "test.vtt"       |  "test_vtt.srt"       |
      |  "test.ssa"       |  "test_ssa.srt"       |
      |  "test.ass"       |  "test_ass.srt"       |
      |  "test.mpl2.txt"  |  "test_mpl2_txt.srt"  |
      |  "test.tmp"       |  "test_tmp.srt"       |
      |  "test.smi"       |  "test_smi.srt"       |
      |  "test.sami"      |  "test_sami.srt"      |
      |  "test.stl"       |  "test_stl.srt"       |
      |  "test.sub"       |  "test_srt.srt"       |
      |  "test.scc"       |  "test_scc.srt"       |
      |  "test.ttml"      |  "test_ttml.vtt"      |
      |  "test.xml"       |  "test_xml.vtt"       |
      |  "test.dfxp"      |  "test_dfxp.vtt"      |
      |  "test.vtt"       |  "test_vtt.ttml"      |
      |  "test.ssa"       |  "test_ssa.ass"       |
      |  "test.ass"       |  "test_ass.ssa"       |
      |  "test.mpl2.txt"  |  "test_mpl2_txt.tmp"  |
      |  "test.tmp"       |  "test_tmp.mpl2.txt"  |
      |  "test.smi"       |  "test_smi.stl"       |
      |  "test.sami"      |  "test_sami.stl"      |
      |  "test.stl"       |  "test_stl.smi"       |
      |  "test.scc"       |  "test_scc.sub"       |

  Scenario: Test conversion from a remote subtitle file
    Given I have a subtitle file "https://raw.githubusercontent.com/baxtree/subaligner/master/tests/subaligner/resource/test.srt"
    When I run the converter with "test_srt.ttml" as the output
    Then a new subtitle file "test_srt.ttml" is generated
