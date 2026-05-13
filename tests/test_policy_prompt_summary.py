from src.policy.context_manager import build_policy_prompt_summary


def test_policy_prompt_summary_reflects_dense_first_guidance():
    summary = build_policy_prompt_summary(
        policy={
            "policy_version": "atlas-test",
            "engine_limits": {"max_segment_seconds": 10.0},
            "annotation": {
                "preferred_density_sec": {"min": 2.0, "max": 5.0},
                "max_atomic_actions": 2,
                "max_label_words": 20,
            },
            "lexicon": {
                "forbidden_verbs": ["inspect", "check"],
                "forbidden_narrative_words": ["then"],
            },
            "behavior": {
                "segment_style": "dense_first",
                "correct_existing_labels_preferred": True,
                "label_task_relevant_pauses_within_segment": True,
            },
        }
    )

    assert "Segment style preference: dense_first." in summary
    assert "Correct existing labels before rewriting from scratch: yes." in summary
    assert "Label task-relevant pauses inside the kept segment: yes." in summary
