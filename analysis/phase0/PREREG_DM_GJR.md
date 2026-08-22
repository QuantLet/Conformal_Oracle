# Pre-registration: Diebold--Mariano, GJR-GARCH vs GJR-GARCH-t, corrected series

Written before running. Standing rule 3.

The two corrected quantile scores are 4.669694 and 4.670076 (x 1e-4), a gap of
0.008%. I expect the Driscoll--Kraay panel-HAC DM test to return a p-value far
above 0.05 -- I would be surprised by anything below 0.5 -- and I expect the sign
of the mean loss differential to be uninformative. If the test returned p < 0.05
on a 0.008% gap I would treat that as evidence of a bug in my variance
estimator, not as a real ranking.
