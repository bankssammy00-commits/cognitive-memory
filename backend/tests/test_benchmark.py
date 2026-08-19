from tests.benchmark_data import CONVERSATION, QUERY


def test_benchmark_dataset():

    assert len(CONVERSATION) == 5

    assert "14 days" in CONVERSATION[0]["text"]
    assert "Alpha" in CONVERSATION[1]["text"]
    assert "Beta" in CONVERSATION[2]["text"]

    assert QUERY == "Which supplier should I consider?"