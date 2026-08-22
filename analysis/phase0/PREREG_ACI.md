# Pre-registration: ACI on the declared 312-pair panel

Written before recomputation. Standing rule 3.

The manuscript claims: "Among single-shot procedures the conformal shift has the
best quantile score (5.14e-4) and the highest green rate (89.1%)."

From reading the printed table I expect:
  E1  ACI green rate 97.4% > conformal static 89.1%  -> the "highest green rate"
      clause FAILS unless "single-shot" legitimately excludes ACI.
  E2  ACI quantile score 5.29e-4 > static 5.14e-4 (worse) -> the "best quantile
      score" clause SURVIVES.
  E3  ACI mean threshold 0.039 vs rolling 0.043 -> ACI is NARROWER, so the
      "wider intervals" claim is false on this panel.
  E4  ACI pi-hat 0.013 against nominal 0.010 -> ACI UNDER-covers, i.e. it is
      further from nominal than the static shift at 0.011.

Therefore I predict ACI does NOT dominate on both axes: it wins the regulatory
axis and loses the scoring axis, and it is the worst of the three on unconditional
coverage. If that holds, S5.7 keeps its conclusion with the green-rate clause
corrected. If instead ACI wins QS as well, the conclusion changes and AE-5 is
conceded in print.
