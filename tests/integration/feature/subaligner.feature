Feature: Subaligner CLI

    @subaligner_1pass
    Scenario: Test subaligner single-stage alignment
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the single-stage alignment on them
        Then a new subtitle file "test_aligned.srt" is generated

    @subaligner_1pass
    Scenario: Test subaligner single-stage alignment with customer output
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the single-stage alignment on them with output "customer_aligned.srt"
        Then a new subtitle file "customer_aligned.srt" is generated

    @subaligner_2pass
    Scenario: Test subaligner dual-stage alignment
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the dual-stage alignment on them
        Then a new subtitle file "test_aligned.srt" is generated

    @subaligner_2pass
    Scenario: Test subaligner dual-stage alignment with customer output
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the dual-stage alignment on them with output "customer_aligned.srt"
        Then a new subtitle file "customer_aligned.srt" is generated

    @subaligner_2pass
    Scenario: Test subaligner dual-stage alignment without stretch
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the dual-stage alignment on them without stretch
        Then a new subtitle file "test_aligned.srt" is generated

    @quality-management
    Scenario: Test exit when single-stage alignment log loss is too high
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        And I set the max log loss to "0.000001"
        When I run the single-stage alignment on them with max loss
        Then it exits with code "22"

    @quality-management
    Scenario: Test exit when dual-stage alignment log loss is too high
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        And I set the max log loss to "0.000001"
        When I run the dual-stage alignment on them with max loss
        Then it exits with code "22"

    @help
    Scenario: Test subaligner single-stage help
        When I run the single-stage command with help
        Then the single-stage help information is displayed

    @help
    Scenario: Test subaligner dual-stage help
        When I run the dual-stage command with help
        Then the dual-stage help information is displayed
