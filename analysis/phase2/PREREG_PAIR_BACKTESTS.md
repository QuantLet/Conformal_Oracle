# Pre-registration — backtests on the constructed pair

Written before `pair_backtests.py` is run, and before any of its output is read.

## What is being recomputed and why

Supplement S.8 prints seven figures for which the repository holds no producer:

    Simulated at T = 500,000 it returns Kupiec p = 0.138 and 0.314,
    Christoffersen independence p = 0.776 and 0.625, Basel Green for both,
    and a dynamic quantile test at p = 0.943 and 0.831 ... at T = 1,500 a
    size-5% one-sided test calibrated on the honest model still rejects the
    alternative with probability 0.079.

`construct_pair.py` writes `pair.npz` and `pair_report.json` and stops there.
`sim.npz` holds 20,000 draws and no statistic. The objects exist, so the figures
are recomputed rather than removed.

## The unit, declared before running

Two exercises, two units.

**Exercise A — the six p-values.** One row is a **simulated path**: T = 500,000
daily observations under one of the two laws, with the forecaster reporting that
law's own 1% threshold. Expected rows: 2 (one per law). Three tests per row, so
six p-values. What varies between rows is the return law only; the reported
threshold moves with it, which is the whole point of the construction.

**Exercise B — the power figure.** One row is a **replication**: an independent
path of T = 1,500 under the mean-ES-matched alternative, with the Z_2 statistic
compared against a critical value calibrated on the honest model. Expected rows:
the replication count, declared in the script. What varies is the seed, over a
fixed pair of laws.

Exercise B needs a law `construct_pair.py` does not build: the mean expected
shortfall enters as a **fifth** equality constraint on the same grid. That LP is
solved in the same script, and its feasibility is reported before its power is.

## The expected result, in both directions, written in advance

### Exercise A

By construction both laws put exactly alpha mass strictly below their own
reported threshold, so the exceedance indicator is Bernoulli(alpha) and
independent over time under **both**. The two null distributions are therefore
not approximately equal but **identical**, and every exceedance-path test has the
same law under P and under Q.

**Expected:** six p-values that look like six draws from U(0,1), and Basel Green
for both paths. The printed set -- 0.138, 0.314, 0.776, 0.625, 0.943, 0.831 --
is consistent with that and so is almost any other set.

**What this exercise can and cannot establish, stated before it runs.** It cannot
be evidence that the tests are weak: that is a theorem about the construction,
not a simulation result, and it is proved by the equality of the two u-processes.
What the simulation checks is that the construction was *implemented* correctly
-- that the discrete law the linear programme returned really does put alpha mass
below q_trunc when sampled, rather than only in the LP's equality constraints.
The supplement must say which of the two it is doing. Reporting six p-values as
though they were the finding overstates them; the finding is the equality above.

**Falsified if:** either path rejects at the 5% level on any of the three tests,
or either lands outside the Basel Green zone. That would mean the sampled law
does not reproduce its own design constraints -- a defect in `construct_pair.py`
or in the sampler, not a property of the tests -- and the paragraph would be
blocked until it is found.

**A p-value near 0 or 1 is not a failure** and is not to be re-seeded away. The
seed is declared in the script and fixed once. If the first draw returns
p = 0.002 on some test, that is reported, with the observation that under a
correct construction it happens 1 time in 500 per test.

### Exercise B

The claim the sentence makes is "necessary, not sufficient": matching mean ES is
a real constraint but does not blind a magnitude-reading test.

**If the rejection probability is materially above the 5% size** -- the printed
0.079 is such a value -- then the reading holds. Z_2 retains power against an
alternative matched on mean ES, so mean-ES matching is not enough to hide a
truncated tail, and the ES backtest is doing something the coverage tests cannot.

**If it is at or below 5%** -- at or under nominal size -- then the reading
fails and must be reversed rather than trimmed. It would say that adding the mean
ES constraint *does* blind the magnitude-reading test as well, which is a
stronger version of the paper's own thesis and strengthens Section 8 rather than
weakening it. The sentence would then read that ES matching is sufficient too,
and "Necessary, not sufficient" would be deleted as false.

**The third outcome, and it is not a fudge:** the power may sit so close to size
that the Monte Carlo error covers the gap. 0.079 against 0.05 at any replication
count under a few thousand is exactly such a case. The script therefore reports a
confidence interval on the rejection frequency beside the point estimate, and if
that interval contains the nominal size the result is stated as **not
distinguishable from size at this replication count**, with the count named. A
power figure whose interval covers its own null is not evidence of power and will
not be printed as though it were.

**Falsified if** the fifth constraint turns out to be infeasible on the grid, in
which case the alternative the sentence describes does not exist and the claim is
removed rather than replaced. This is the R6 failure mode -- an infeasibility
verdict from a correct model on a malformed support -- so infeasibility is
reported together with the grid range, and the grid is widened once before the
verdict is accepted.

## What would falsify the exercise itself

If the sampled exceedance rate under either law differs from alpha by more than
Monte Carlo error at T = 500,000, the construction is not delivering what
`pair_report.json` claims and both exercises are BLOCKED rather than reported.
