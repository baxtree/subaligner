Feature: Subaligner CLI

    Scenario: Test subaligner single-stage alignment
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the single-stage alignment on them
        Then a new subtitle file "test_aligned.srt" is generated
        
    Scenario: Test subaligner single-stage alignment with customer output
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the single-stage alignment on them with output "customer_aligned.srt"
        Then a new subtitle file "customer_aligned.srt" is generated
        
    Scenario: Test subaligner dual-stage alignment
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the dual-stage alignment on them
        Then a new subtitle file "test_aligned.srt" is generated
        
    Scenario: Test subaligner dual-stage alignment with customer output
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the dual-stage alignment on them with output "customer_aligned.srt"
        Then a new subtitle file "customer_aligned.srt" is generated
        
    Scenario: Test subaligner dual-stage alignment without stretch
        Given I have a video file "test.mp4"
        And I have a subtitle file "test.srt"
        When I run the dual-stage alignment on them without stretch
        Then a new subtitle file "test_aligned.srt" is generated
