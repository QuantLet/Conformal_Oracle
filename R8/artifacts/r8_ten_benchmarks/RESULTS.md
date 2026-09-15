# Benchmarks on the ten-model support (measured)

{
  "assets": 24,
  "test_observations": 36252,
  "alpha": 0.01,
  "warmup": 512,
  "calibration_fraction": 0.7,
  "rolling_window": 250,
  "calibration_inside_fit_min": 1629,
  "calibration_inside_fit_max": 4332,
  "calibration_inside_fit_share_min": 0.9141414141414141,
  "calibration_inside_fit_share_max": 0.9656709763709318,
  "test_after_fit_all": true,
  "shared_series": 21,
  "shared_support_identical": 21,
  "replaced_series": [
    "GLD",
    "UNG",
    "USO"
  ],
  "pre_test_min": 153,
  "pre_test_max": 154,
  "elapsed_seconds": 0.42405008300011104
}

     model     method  QS_x10000  pi_mean    width  kupiec_rejections  independence_rejections  conditional_rejections  scaled_green
CAViaR-SAV        Raw   4.632401 0.010117 0.036271                  8                        5                       8            22
CAViaR-SAV     Static   4.621523 0.009762 0.036150                  8                        4                       9            22
CAViaR-SAV Rolling250   4.799646 0.008079 0.039779                  1                        3                       2            24
 CAViaR-AS        Raw   4.665077 0.010987 0.035527                  9                        3                       6            19
 CAViaR-AS     Static   4.637200 0.010820 0.035280                  7                        2                       6            20
 CAViaR-AS Rolling250   4.839596 0.008255 0.039875                  1                        4                       3            24
     GAS-t        Raw   4.709883 0.013085 0.034675                  9                        7                      11            17
     GAS-t     Static   4.687791 0.010147 0.035937                  6                        5                       9            21
     GAS-t Rolling250   4.920757 0.008429 0.040814                  0                        3                       2            24
   EVT-POT        Raw   4.540368 0.013844 0.033115                  5                        4                       6            22
       FHS        Raw   4.602000 0.015132 0.032314                 12                        6                      12            15

  asset  n_returns  n_eligible  n_cal  n_test first_test dynamic_fit_last  calibration_dates_inside_fit  test_after_fit  dedicated_pre_test_forecasts  reference_support_identical
 ASX200       6736        6224   4356    1868 2019-04-12       2018-09-04                          4203            True                           153                         True
 AUDUSD       5277        4765   3335    1430 2021-03-02       2020-07-28                          3181            True                           154                         True
BOVESPA       6606        6094   4265    1829 2019-04-29       2018-09-06                          4112            True                           153                         True
    BTC       4366        3854   2697    1157 2023-07-02       2023-01-29                          2544            True                           153                         True
   CBU0       3878        3366   2356    1010 2022-08-26       2022-01-14                          2202            True                           154                         True
   DJCI       3059        2547   1782     765 2023-08-15       2023-01-03                          1629            True                           153                         True
    ETH       3217        2705   1893     812 2024-06-11       2024-01-08                          1739            True                           154                         True
 EURUSD       5901        5389   3772    1617 2020-06-12       2019-11-08                          3618            True                           154                         True
   FCHI       6813        6301   4410    1891 2019-04-11       2018-09-04                          4257            True                           153                         True
FTSE100       6733        6221   4354    1867 2019-04-05       2018-08-29                          4201            True                           153                         True
 GBPUSD       5913        5401   3780    1621 2020-06-08       2019-11-05                          3627            True                           153                         True
  GDAXI       6769        6257   4379    1878 2019-04-10       2018-08-29                          4226            True                           153                         True
    GLD       5478        4966   3476    1490 2020-09-24       2020-02-13                          3322            True                           154                        False
    HSI       6567        6055   4238    1817 2019-04-11       2018-08-23                          4084            True                           154                         True
   IBGL       4711        4199   2939    1260 2021-08-27       2021-01-18                          2785            True                           154                         True
   ICLN       4573        4061   2842    1219 2021-10-21       2021-03-15                          2689            True                           153                         True
  NIFTY       4649        4137   2895    1242 2021-08-23       2021-01-06                          2742            True                           153                         True
 NIKKEI       6529        6017   4211    1806 2019-04-04       2018-08-15                          4058            True                           153                         True
  SP500       6704        6192   4334    1858 2019-04-10       2018-08-27                          4180            True                           154                         True
  STOXX       5618        5106   3574    1532 2020-07-30       2019-12-11                          3420            True                           154                         True
    TLT       6060        5548   3883    1665 2020-01-15       2019-06-06                          3730            True                           153                         True
    UNG       4873        4361   3052    1309 2021-06-15       2020-11-02                          2899            True                           153                        False
 USDJPY       6921        6409   4486    1923 2019-04-10       2018-09-05                          4332            True                           154                         True
    USO       5129        4617   3231    1386 2021-02-24       2020-07-15                          3078            True                           153                        False
