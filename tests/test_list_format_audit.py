from scripts.audit_list_format import audit_rows, marker_count, request_kind


def test_narrow_request_grammar_skips_ambiguous_and_negative():
    assert request_kind("Please use bullet points.") == "bulleted"
    assert request_kind("Make a numbered list.") == "numbered"
    assert request_kind("Use a numbered list or a bulleted list.") is None
    assert request_kind("Don’t use a numbered list.") is None
    assert request_kind("list three options") is None


def test_marker_counts_exclude_fenced_code_and_midline_numbers():
    fence = "```"
    assert marker_count(f"1) first\n{fence}python\n2. example\n{fence}\n 3. end", "numbered") == 2
    assert marker_count("There were 2.5 of them.\n- one\n • two\n*bold*", "bulleted") == 2
    assert marker_count("  1. a\n  1. b\n- c", "numbered") == 2


def test_count_adjacent_turns_without_carrying_over_previous_request():
    messages = [
        {"role": role, "content": text} for role, text in [
            ("system", "Please use bullet points."), ("user", "Use a numbered list."),
            ("assistant", "Paragraph"), ("user", "Thanks"), ("assistant", "1. one\n2. two"),
            ("user", "Respond in bullet points"), ("assistant", "- one"),
        ]
    ]
    result = audit_rows([{"messages": messages}])
    assert result["adjacent_user_assistant_pairs"] == 3
    assert result["dialogues_with_any_matched_turn"] == 1
    assert result["detected_turns"]["numbered"] == {"zero": 1, "one": 0, "two_or_more": 0}
    assert result["detected_turns"]["bulleted"]["one"] == 1
