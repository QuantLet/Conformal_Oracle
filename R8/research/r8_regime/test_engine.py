import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
import engine as e


@pytest.mark.parametrize('kind', ['normal', 't5'])
@pytest.mark.parametrize('alpha', [.01, .05])
def test_integrated_risk(kind, alpha):
    s = .027
    q = s*e.unit_quantile(kind, alpha)+.013
    density = (lambda x: stats.norm.pdf(x, scale=s)) if kind == 'normal' else (lambda x: stats.t.pdf(x/(s*np.sqrt(3/5)), 5)/(s*np.sqrt(3/5)))
    numeric = quad(lambda x: (1-alpha)*(q-x)*density(x), -np.inf, q, epsabs=1e-12)[0]
    numeric += quad(lambda x: alpha*(x-q)*density(x), q, np.inf, epsabs=1e-12)[0]
    risk, p = e.expected(kind, q, s, alpha)
    assert abs(risk-numeric) < 1e-11
    assert 0 < p < 1
    optimal, nominal = e.expected(kind, s*e.unit_quantile(kind, alpha), s, alpha)
    assert optimal < risk and abs(nominal-alpha) < 1e-12


@pytest.mark.parametrize('alpha', [.01, .05])
def test_cp_rank_and_scalar_dtaci(alpha):
    rng = np.random.default_rng(821)
    y = rng.standard_normal((2, 1530))
    # Tied scores exercise inverse-CDF corner cases in the reused method.
    q = np.full_like(y, -1.7)
    y[0, ::3] = 0.
    result = e.policies(y, q, np.ones_like(y), alpha)
    for row in range(2):
        old = e.original.dtaci((q-y)[row, 500:], q[row, 500:], alpha=alpha)
        assert np.allclose(result['experts'][row], old['predictions'][750:], rtol=0, atol=2e-13)
        assert np.allclose(result['probabilities'][row], old['probabilities'][750:], rtol=0, atol=2e-13)
        for w in e.WINDOWS:
            for t in (1250, 1500, 1529):
                c = e.original.cp((q-y)[row, t-w:t], alpha)
                assert result['prediction'][row, t-1250, 2+e.WINDOWS.index(w)] == q[row, t]-c


def test_future_and_current_outcome_cannot_change_forecasts():
    x = e.innovations('t5', [765, 766])
    other = x.copy()
    other[:, e.BURN+1600:] *= -4
    for scenario in ('reverses', 'jump_ewma'):
        y, q, s, _ = e.environment(x, scenario, 't5', .01)
        yy, qq, ss, _ = e.environment(other, scenario, 't5', .01)
        a = e.policies(y, q, s, .01)
        b = e.policies(yy, qq, ss, .01)
        for field in ('prediction', 'experts', 'probabilities', 'states'):
            assert np.array_equal(a[field][:, :351], b[field][:, :351]), field
        for field in ('gate', 'selected', 'validation_loss', 'static_shift'):
            assert np.array_equal(a[field], b[field]), field


def test_break_and_ewma_independent_scalar_recursion():
    x = e.innovations('normal', [711])
    y, q, s, actual = e.environment(x, 'jump_ewma', 'normal', .05)
    variance = e.V0**2
    for t in range(x.shape[1]):
        if t >= e.BURN:
            assert np.isclose(s[0, t-e.BURN], np.sqrt(variance), rtol=2e-15, atol=0)
        true_scale = e.V0*(2 if t >= e.BURN+e.BREAK else 1)
        r = x[0, t]*true_scale
        variance = .94*variance+.06*r*r
    assert np.all(actual[:, :e.BREAK] == e.V0)
    assert np.all(actual[:, e.BREAK:] == 2*e.V0)
    assert np.array_equal(q, s*stats.norm.ppf(.05))


def test_oracle_scale_normalisation_equivariance():
    x = e.innovations('normal', [799])
    outputs = []
    for scenario in ('correct', 'jump_oracle'):
        y, q, s, sigma = e.environment(x, scenario, 'normal', .01)
        a = e.policies(y, q, s, .01)
        outputs.append((a['prediction']/sigma[:, e.START:, None], a))
    assert np.allclose(outputs[0][0][:, :, 6:8], outputs[1][0][:, :, 6:8], atol=2e-14, rtol=0)
    assert np.array_equal(outputs[0][1]['selected'], outputs[1][1]['selected'])


def test_mixture_loss_is_not_mean_quantile_loss():
    q = np.array([[[ -3., -.2 ]]]); w = np.array([[[.5, .5]]])
    risk, _ = e.expected('normal', q, 1., .01)
    mean_risk, _ = e.expected('normal', (q*w).sum(axis=2), 1., .01)
    assert (risk*w).sum() > mean_risk.item()+.05


def test_window_maximum_and_boundary_kupiec():
    s = np.arange(125.)[None, :]
    assert e.cp_rows(s, .01).item() == 124
    hits = np.zeros((2, 250), dtype=bool)
    hits[1, :3] = True
    assert np.array_equal(e.kupiec_reject(hits, .01), [True, False])
