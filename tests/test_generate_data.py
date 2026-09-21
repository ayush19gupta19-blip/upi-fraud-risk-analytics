from src.generate_data import create_dataset


def test_dataset_has_expected_shape_and_columns():
    data = create_dataset(number_of_transactions=100, seed=7)

    assert len(data) == 100
    assert {"transaction_id", "amount", "risk_flag", "risk_scenario"}.issubset(data.columns)
    assert data["transaction_id"].is_unique


def test_dataset_marks_some_review_scenarios():
    data = create_dataset(number_of_transactions=1_000, seed=7)

    assert data["risk_flag"].sum() > 0
    assert "normal" in set(data["risk_scenario"])
