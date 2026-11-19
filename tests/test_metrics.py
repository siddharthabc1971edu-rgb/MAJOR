import numpy as np

from credit_audit.audit import conformal_p_values, routing_table, youden_threshold
from credit_audit.metrics import expected_calibration_error


def test_ece_weights_bin_miscalibration_by_population_share():
    assert np.isclose(
        expected_calibration_error([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], bins=2),
        0.1,
    )


def test_youden_threshold_separates_perfect_predictions():
    threshold = youden_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 <= threshold <= 0.8


def test_conformal_p_values_are_finite_and_nonzero():
    values = conformal_p_values(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]),
        np.array([0.15, 0.85]), np.array([0, 1]),
    )
    assert values.shape == (2,)
    assert np.all((values > 0) & (values <= 1))


def test_routing_reports_coverage_and_review_risk():
    table, selected = routing_table(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.9, 0.4, 0.6]),
        0.5,
        np.array([0.9, 0.7, 0.75, 0.95]),
    )
    assert {"policy", "coverage", "selective_error", "composite_review_need"} <= set(table.columns)
    assert table["coverage"].between(0, 1).all()
    assert selected[("P2 margin + stability", 0.0)].sum() < selected[("P1 margin", 0.0)].sum()
