# 97.5% ES and capital component (measured)

{
  "pairs": 168,
  "assets": 24,
  "alpha": 0.025,
  "es_draws": 25,
  "es_window_points": 6,
  "ES975_raw_pair_equal": 0.03291095543844623,
  "ES975_static_pair_equal": 0.038715275806270114,
  "ES_ratio_pair_equal": 1.1729816328425888,
  "VaR1_ratio_pair_equal": 1.1832441074674018,
  "component_raw": 0.056643142666316926,
  "component_static": 0.05872242178864956,
  "component_ratio": 1.0367083997189492,
  "component_raw_hi": 0.05944973488843782,
  "component_static_hi": 0.05943688067581838,
  "component_ratio_hi": 0.9997837801523662,
  "pairs_component_falls": 64,
  "pairs_component_falls_hi": 81,
  "zones_raw": {
    "Yellow": 62,
    "Red": 55,
    "Green": 51
  },
  "zones_static": {
    "Green": 149,
    "Yellow": 19
  },
  "elapsed_seconds": 9.621239625004819
}

      model     law  assets  VaR1_raw  VaR1_static  ES975_raw  ES975_static  ES_ratio  VaR_ratio  component_raw  component_static  component_ratio  component_raw_hi  component_static_hi  component_ratio_hi  pairs_component_falls  pairs_component_falls_hi
 Moirai-1.1   draws      24  0.034494     0.038883   0.038296      0.042685  1.104053   1.115903       0.062706          0.064571         1.039604          0.066367             0.065169            1.006973                      7                         8
  Lag-Llama   draws      24  0.029496     0.040738   0.033338      0.044579  1.329543   1.373984       0.066675          0.067409         1.008182          0.066675             0.068004            1.020310                     14                        14
  GJR-GARCH  normal      24  0.029251     0.034820   0.029396      0.034965  1.179620   1.180515       0.051465          0.053115         1.021281          0.053820             0.053849            0.996753                     10                        12
GJR-GARCH-t student      24  0.032073     0.035309   0.033330      0.036566  1.103524   1.107127       0.052998          0.055312         1.041471          0.055808             0.055821            0.999111                      8                        10
    GARCH-N  normal      24  0.029464     0.035753   0.029611      0.035900  1.197749   1.198740       0.051623          0.054533         1.042567          0.054327             0.055283            1.013114                      6                        12
   Hist-Sim  window      24  0.034068     0.038044   0.036746      0.040722  1.111641   1.120789       0.059286          0.061404         1.045812          0.063516             0.061756            0.999298                     10                        11
       EWMA  normal      24  0.029515     0.035444   0.029661      0.035589  1.184741   1.185651       0.051749          0.054714         1.032965          0.055634             0.056176            0.994005                      9                        14
