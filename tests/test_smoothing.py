import math

import numpy as np
import pytest

from server.smoothing import Smoother

A = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 1.0, 0.0])


def test_first_update_is_the_input():
    np.testing.assert_allclose(Smoother(1.5).update([0.2, 0.3, 0.5], 1.0), [0.2, 0.3, 0.5])


@pytest.mark.parametrize("n", [3, 6, 20])
def test_output_is_always_a_distribution(n):
    rng = np.random.default_rng(n)
    s = Smoother(1.5)
    for k in range(50):
        v = s.update(rng.dirichlet(np.ones(n)), t=0.5 * k)
        assert v.shape == (n,)
        assert v.sum() == pytest.approx(1.0, abs=1e-12)
        assert v.min() >= 0 and v.max() <= 1


def test_constant_input_converges():
    s = Smoother(1.5)
    s.update(A, 0)
    for k in range(1, 40):
        v = s.update(B, k * 0.5)
    np.testing.assert_allclose(v, B, atol=1e-5)


def test_step_change_is_63_percent_visible_after_tau():
    s = Smoother(1.5)
    s.update(A, 0.0)
    assert s.update(B, 1.5)[1] == pytest.approx(1 - math.exp(-1))


def test_update_rate_does_not_change_the_result():
    every_second, every_half_second = Smoother(1.5), Smoother(1.5)
    every_second.update(A, 0.0)
    every_half_second.update(A, 0.0)
    for t in (0.5, 1.0, 1.5, 2.0):
        every_half_second.update(B, t)
    for t in (1.0, 2.0):
        every_second.update(B, t)
    np.testing.assert_allclose(every_second.value, every_half_second.value, atol=1e-12)


def test_long_pause_restarts_from_new_prediction():
    s = Smoother(1.5)
    s.update(A, 0.0)
    np.testing.assert_allclose(s.update(B, 30.0), B, atol=1e-6)


def test_damps_flicker():
    """Raw output swinging 0.3 <-> 0.7 every update becomes a small wobble."""
    s = Smoother(1.5)
    values = [s.update([0.7, 0.3] if k % 2 == 0 else [0.3, 0.7], t=float(k))[0] for k in range(30)]
    assert max(abs(v - 0.5) for v in values[10:]) < 0.1   # raw swings 0.2 around 0.5


def test_rejects_time_going_backwards_and_class_count_change():
    s = Smoother(1.5)
    s.update(A, 2.0)
    with pytest.raises(ValueError, match="backwards"):
        s.update(B, 1.0)
    with pytest.raises(ValueError, match="class count"):
        s.update([0.5, 0.5], 3.0)


def test_reset():
    s = Smoother(1.5)
    s.update(A, 0.0)
    s.reset()
    assert s.value is None
    np.testing.assert_allclose(s.update(B, 0.0), B)
