## Spatial blocks

| corr_limit | range_km | capped | block_km | buffer_km | groups |
|---|---|---|---|---|---|
| 0.200 | 40.000 | False | 80.000 | 20.000 | 4 |

## Model ladder, pr_auc

| cv | target | h | subset | n | base_rate | climatology: baseline | persistence: baseline | S: gbm | M1: logreg | M1: gbm | M1_physical: logreg | M1_physical: gbm | M2: gbm | M2_physical: gbm | M3: logreg | M3: gbm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strict | occurrence | 1 | all | 31832 | 0.151 | 0.410 | 0.195 | 0.401 | 0.180 | 0.242 | 0.188 | 0.250 | 0.411 | 0.416 | 0.318 | 0.443 |
| strict | occurrence | 1 | cropland | 8127 | 0.155 | 0.438 | 0.194 | 0.378 | 0.156 | 0.210 | 0.186 | 0.226 | 0.397 | 0.402 | 0.268 | 0.426 |
| strict | occurrence | 1 | cropland_harvest | 3131 | 0.220 | 0.447 | 0.244 | 0.406 | 0.218 | 0.233 | 0.240 | 0.256 | 0.415 | 0.429 | 0.302 | 0.431 |
| strict | occurrence | 1 | harvest | 12293 | 0.234 | 0.471 | 0.267 | 0.457 | 0.249 | 0.265 | 0.249 | 0.276 | 0.442 | 0.450 | 0.375 | 0.471 |
| strict | persistence | 1 | all | 11649 | 0.617 | 0.853 | 0.672 | 0.826 | 0.668 | 0.713 | 0.665 | 0.708 | 0.836 | 0.832 | 0.832 | 0.858 |
| strict | persistence | 1 | cropland | 2925 | 0.613 | 0.873 | 0.652 | 0.849 | 0.632 | 0.722 | 0.640 | 0.707 | 0.858 | 0.846 | 0.844 | 0.870 |
| strict | persistence | 1 | cropland_harvest | 1474 | 0.613 | 0.882 | 0.659 | 0.870 | 0.676 | 0.735 | 0.649 | 0.728 | 0.879 | 0.875 | 0.866 | 0.883 |
| strict | persistence | 1 | harvest | 5602 | 0.633 | 0.869 | 0.689 | 0.854 | 0.702 | 0.735 | 0.674 | 0.728 | 0.856 | 0.854 | 0.862 | 0.874 |

## Model ladder, brier

| cv | target | h | subset | n | base_rate | climatology: baseline | persistence: baseline | S: gbm | M1: logreg | M1: gbm | M1_physical: logreg | M1_physical: gbm | M2: gbm | M2_physical: gbm | M3: logreg | M3: gbm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strict | occurrence | 1 | all | 31832 | 0.151 | 0.110 | 0.140 | 0.110 | 0.177 | 0.126 | 0.149 | 0.124 | 0.110 | 0.108 | 0.151 | 0.111 |
| strict | occurrence | 1 | cropland | 8127 | 0.155 | 0.109 | 0.143 | 0.115 | 0.199 | 0.133 | 0.152 | 0.131 | 0.114 | 0.113 | 0.162 | 0.115 |
| strict | occurrence | 1 | cropland_harvest | 3131 | 0.220 | 0.152 | 0.204 | 0.158 | 0.264 | 0.184 | 0.222 | 0.177 | 0.158 | 0.155 | 0.226 | 0.165 |
| strict | occurrence | 1 | harvest | 12293 | 0.234 | 0.159 | 0.217 | 0.163 | 0.247 | 0.189 | 0.236 | 0.184 | 0.165 | 0.161 | 0.211 | 0.167 |
| strict | persistence | 1 | all | 11649 | 0.617 | 0.197 | 0.280 | 0.185 | 0.307 | 0.244 | 0.319 | 0.247 | 0.185 | 0.188 | 0.198 | 0.175 |
| strict | persistence | 1 | cropland | 2925 | 0.613 | 0.186 | 0.275 | 0.175 | 0.341 | 0.254 | 0.324 | 0.244 | 0.180 | 0.183 | 0.192 | 0.167 |
| strict | persistence | 1 | cropland_harvest | 1474 | 0.613 | 0.182 | 0.277 | 0.171 | 0.355 | 0.244 | 0.343 | 0.236 | 0.164 | 0.164 | 0.187 | 0.161 |
| strict | persistence | 1 | harvest | 5602 | 0.633 | 0.197 | 0.267 | 0.179 | 0.331 | 0.239 | 0.350 | 0.241 | 0.177 | 0.178 | 0.195 | 0.173 |

## All metrics (threshold metrics use a cutoff chosen on the other test years)

| cv | h | target | subset | config | model | n | base_rate | accuracy | always_no_accuracy | precision | recall | f1 | balanced_accuracy | false_alarm_rate | roc_auc | gini | pr_auc | brier | brier_skill_vs_clim | log_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strict | 1 | occurrence | all | M1 | gbm | 31832 | 0.151 | 0.577 | 0.849 | 0.207 | 0.635 | 0.312 | 0.601 | 0.433 | 0.664 | 0.327 | 0.242 | 0.126 | -0.140 | 0.416 |
| strict | 1 | occurrence | all | M1 | logreg | 31832 | 0.151 | 0.257 | 0.849 | 0.151 | 0.847 | 0.256 | 0.499 | 0.848 | 0.549 | 0.098 | 0.180 | 0.177 | -0.609 | 0.612 |
| strict | 1 | occurrence | all | M1_physical | gbm | 31832 | 0.151 | 0.605 | 0.849 | 0.226 | 0.668 | 0.338 | 0.631 | 0.406 | 0.672 | 0.343 | 0.250 | 0.124 | -0.127 | 0.410 |
| strict | 1 | occurrence | all | M1_physical | logreg | 31832 | 0.151 | 0.369 | 0.849 | 0.156 | 0.722 | 0.257 | 0.514 | 0.694 | 0.567 | 0.133 | 0.188 | 0.149 | -0.349 | 0.551 |
| strict | 1 | occurrence | all | M2 | gbm | 31832 | 0.151 | 0.773 | 0.849 | 0.344 | 0.557 | 0.425 | 0.684 | 0.189 | 0.779 | 0.557 | 0.411 | 0.110 | 0.004 | 0.361 |
| strict | 1 | occurrence | all | M2 | logreg | 31832 | 0.151 | 0.680 | 0.849 | 0.240 | 0.518 | 0.328 | 0.613 | 0.291 | 0.665 | 0.330 | 0.267 | 0.151 | -0.367 | 0.503 |
| strict | 1 | occurrence | all | M2_physical | gbm | 31832 | 0.151 | 0.807 | 0.849 | 0.395 | 0.524 | 0.450 | 0.691 | 0.143 | 0.786 | 0.572 | 0.416 | 0.108 | 0.016 | 0.356 |
| strict | 1 | occurrence | all | M3 | gbm | 31832 | 0.151 | 0.796 | 0.849 | 0.380 | 0.556 | 0.452 | 0.697 | 0.161 | 0.785 | 0.571 | 0.443 | 0.111 | -0.005 | 0.451 |
| strict | 1 | occurrence | all | M3 | logreg | 31832 | 0.151 | 0.731 | 0.849 | 0.280 | 0.501 | 0.360 | 0.636 | 0.229 | 0.708 | 0.415 | 0.318 | 0.151 | -0.371 | 0.527 |
| strict | 1 | occurrence | all | climatology | baseline | 31832 | 0.151 | 0.799 | 0.849 | 0.384 | 0.547 | 0.451 | 0.695 | 0.156 | 0.794 | 0.587 | 0.410 | 0.110 | 0.000 | 0.470 |
| strict | 1 | occurrence | all | persistence | baseline | 31832 | 0.151 | 0.814 | 0.849 | 0.335 | 0.234 | 0.276 | 0.576 | 0.083 | 0.576 | 0.152 | 0.195 | 0.140 | -0.269 | 1.667 |
| strict | 1 | occurrence | cropland_harvest | M1 | gbm | 3131 | 0.220 | 0.407 | 0.780 | 0.245 | 0.817 | 0.377 | 0.554 | 0.709 | 0.548 | 0.096 | 0.233 | 0.184 | -0.211 | 0.584 |
| strict | 1 | occurrence | cropland_harvest | M1 | logreg | 3131 | 0.220 | 0.274 | 0.780 | 0.200 | 0.768 | 0.318 | 0.451 | 0.865 | 0.490 | -0.019 | 0.218 | 0.264 | -0.738 | 0.880 |
| strict | 1 | occurrence | cropland_harvest | M1_physical | gbm | 3131 | 0.220 | 0.406 | 0.780 | 0.242 | 0.797 | 0.371 | 0.546 | 0.704 | 0.570 | 0.141 | 0.256 | 0.177 | -0.165 | 0.553 |
| strict | 1 | occurrence | cropland_harvest | M1_physical | logreg | 3131 | 0.220 | 0.314 | 0.780 | 0.214 | 0.792 | 0.337 | 0.486 | 0.821 | 0.540 | 0.081 | 0.240 | 0.222 | -0.460 | 0.787 |
| strict | 1 | occurrence | cropland_harvest | M2 | gbm | 3131 | 0.220 | 0.605 | 0.780 | 0.309 | 0.643 | 0.418 | 0.619 | 0.405 | 0.699 | 0.398 | 0.415 | 0.158 | -0.038 | 0.499 |
| strict | 1 | occurrence | cropland_harvest | M2 | logreg | 3131 | 0.220 | 0.568 | 0.780 | 0.275 | 0.591 | 0.376 | 0.576 | 0.439 | 0.588 | 0.176 | 0.262 | 0.232 | -0.524 | 0.767 |
| strict | 1 | occurrence | cropland_harvest | M2_physical | gbm | 3131 | 0.220 | 0.714 | 0.780 | 0.389 | 0.521 | 0.445 | 0.645 | 0.231 | 0.701 | 0.402 | 0.429 | 0.155 | -0.016 | 0.489 |
| strict | 1 | occurrence | cropland_harvest | M3 | gbm | 3131 | 0.220 | 0.649 | 0.780 | 0.350 | 0.694 | 0.465 | 0.665 | 0.363 | 0.724 | 0.447 | 0.431 | 0.165 | -0.083 | 0.647 |
| strict | 1 | occurrence | cropland_harvest | M3 | logreg | 3131 | 0.220 | 0.614 | 0.780 | 0.311 | 0.623 | 0.415 | 0.617 | 0.389 | 0.635 | 0.271 | 0.302 | 0.226 | -0.487 | 0.800 |
| strict | 1 | occurrence | cropland_harvest | climatology | baseline | 3131 | 0.220 | 0.719 | 0.780 | 0.403 | 0.582 | 0.477 | 0.670 | 0.243 | 0.743 | 0.485 | 0.447 | 0.152 | 0.000 | 0.637 |
| strict | 1 | occurrence | cropland_harvest | persistence | baseline | 3131 | 0.220 | 0.220 | 0.780 | 0.220 | 1.000 | 0.361 | 0.500 | 1.000 | 0.546 | 0.091 | 0.244 | 0.204 | -0.343 | 2.426 |
| strict | 1 | persistence | all | M1 | gbm | 11649 | 0.617 | 0.624 | 0.383 | 0.624 | 0.985 | 0.764 | 0.513 | 0.959 | 0.623 | 0.245 | 0.713 | 0.244 | -0.239 | 0.687 |
| strict | 1 | persistence | all | M1 | logreg | 11649 | 0.617 | 0.618 | 0.383 | 0.620 | 0.982 | 0.760 | 0.506 | 0.971 | 0.561 | 0.122 | 0.668 | 0.307 | -0.555 | 0.863 |
| strict | 1 | persistence | all | M1_physical | gbm | 11649 | 0.617 | 0.620 | 0.383 | 0.620 | 0.992 | 0.763 | 0.506 | 0.979 | 0.613 | 0.225 | 0.708 | 0.247 | -0.251 | 0.690 |
| strict | 1 | persistence | all | M1_physical | logreg | 11649 | 0.617 | 0.618 | 0.383 | 0.618 | 0.998 | 0.763 | 0.501 | 0.996 | 0.557 | 0.115 | 0.665 | 0.319 | -0.616 | 0.897 |
| strict | 1 | persistence | all | M2 | gbm | 11649 | 0.617 | 0.718 | 0.383 | 0.719 | 0.893 | 0.797 | 0.665 | 0.564 | 0.783 | 0.565 | 0.836 | 0.185 | 0.064 | 0.551 |
| strict | 1 | persistence | all | M2 | logreg | 11649 | 0.617 | 0.645 | 0.383 | 0.641 | 0.965 | 0.771 | 0.547 | 0.870 | 0.687 | 0.373 | 0.762 | 0.254 | -0.287 | 0.772 |
| strict | 1 | persistence | all | M2_physical | gbm | 11649 | 0.617 | 0.713 | 0.383 | 0.713 | 0.895 | 0.794 | 0.657 | 0.580 | 0.776 | 0.551 | 0.832 | 0.188 | 0.046 | 0.559 |
| strict | 1 | persistence | all | M3 | gbm | 11649 | 0.617 | 0.751 | 0.383 | 0.749 | 0.896 | 0.816 | 0.706 | 0.484 | 0.812 | 0.625 | 0.858 | 0.175 | 0.115 | 0.583 |
| strict | 1 | persistence | all | M3 | logreg | 11649 | 0.617 | 0.706 | 0.383 | 0.707 | 0.894 | 0.790 | 0.649 | 0.596 | 0.780 | 0.559 | 0.832 | 0.198 | -0.005 | 0.619 |
| strict | 1 | persistence | all | climatology | baseline | 11649 | 0.617 | 0.750 | 0.383 | 0.740 | 0.919 | 0.820 | 0.699 | 0.522 | 0.810 | 0.621 | 0.853 | 0.197 | 0.000 | 0.693 |
| strict | 1 | persistence | all | persistence | baseline | 11649 | 0.617 | 0.617 | 0.383 | 0.617 | 1.000 | 0.763 | 0.500 | 1.000 | 0.597 | 0.193 | 0.672 | 0.280 | -0.421 | 2.488 |
| strict | 1 | persistence | cropland_harvest | M1 | gbm | 1474 | 0.613 | 0.615 | 0.387 | 0.615 | 0.993 | 0.760 | 0.504 | 0.986 | 0.624 | 0.248 | 0.735 | 0.244 | -0.344 | 0.684 |
| strict | 1 | persistence | cropland_harvest | M1 | logreg | 1474 | 0.613 | 0.638 | 0.387 | 0.638 | 0.948 | 0.762 | 0.547 | 0.854 | 0.590 | 0.180 | 0.676 | 0.355 | -0.953 | 1.016 |
| strict | 1 | persistence | cropland_harvest | M1_physical | gbm | 1474 | 0.613 | 0.640 | 0.387 | 0.637 | 0.961 | 0.766 | 0.546 | 0.870 | 0.620 | 0.241 | 0.728 | 0.236 | -0.298 | 0.664 |
| strict | 1 | persistence | cropland_harvest | M1_physical | logreg | 1474 | 0.613 | 0.615 | 0.387 | 0.617 | 0.982 | 0.758 | 0.507 | 0.968 | 0.542 | 0.085 | 0.649 | 0.343 | -0.890 | 0.963 |
| strict | 1 | persistence | cropland_harvest | M2 | gbm | 1474 | 0.613 | 0.777 | 0.387 | 0.790 | 0.867 | 0.827 | 0.750 | 0.367 | 0.829 | 0.657 | 0.879 | 0.164 | 0.096 | 0.501 |
| strict | 1 | persistence | cropland_harvest | M2 | logreg | 1474 | 0.613 | 0.672 | 0.387 | 0.666 | 0.930 | 0.777 | 0.596 | 0.739 | 0.705 | 0.409 | 0.782 | 0.282 | -0.553 | 0.856 |
| strict | 1 | persistence | cropland_harvest | M2_physical | gbm | 1474 | 0.613 | 0.759 | 0.387 | 0.773 | 0.861 | 0.814 | 0.729 | 0.402 | 0.826 | 0.651 | 0.875 | 0.164 | 0.097 | 0.500 |
| strict | 1 | persistence | cropland_harvest | M3 | gbm | 1474 | 0.613 | 0.780 | 0.387 | 0.790 | 0.872 | 0.829 | 0.753 | 0.367 | 0.841 | 0.682 | 0.883 | 0.161 | 0.112 | 0.557 |
| strict | 1 | persistence | cropland_harvest | M3 | logreg | 1474 | 0.613 | 0.744 | 0.387 | 0.748 | 0.879 | 0.808 | 0.705 | 0.470 | 0.814 | 0.628 | 0.866 | 0.187 | -0.029 | 0.584 |
| strict | 1 | persistence | cropland_harvest | climatology | baseline | 1474 | 0.613 | 0.773 | 0.387 | 0.760 | 0.919 | 0.832 | 0.730 | 0.460 | 0.842 | 0.685 | 0.882 | 0.182 | 0.000 | 0.643 |
| strict | 1 | persistence | cropland_harvest | persistence | baseline | 1474 | 0.613 | 0.613 | 0.387 | 0.613 | 1.000 | 0.760 | 0.500 | 1.000 | 0.578 | 0.156 | 0.659 | 0.277 | -0.523 | 2.169 |

## Cutoff sweep (strict): alarm budgets and fixed probability cutoffs

| h | target | subset | model | rule | precision | recall | f1 | false_alarm_rate | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 1 | occurrence | all | clim | top 5% flagged | 0.565 | 0.207 | 0.303 | 0.028 | 0.856 |
| 1 | occurrence | all | clim | top 10% flagged | 0.481 | 0.337 | 0.396 | 0.065 | 0.845 |
| 1 | occurrence | all | clim | top 20% flagged | 0.378 | 0.568 | 0.454 | 0.166 | 0.794 |
| 1 | occurrence | all | clim | top 30% flagged | 0.335 | 0.667 | 0.446 | 0.235 | 0.750 |
| 1 | occurrence | all | clim | p >= 0.1 | 0.271 | 0.836 | 0.410 | 0.400 | 0.636 |
| 1 | occurrence | all | clim | p >= 0.2 | 0.336 | 0.667 | 0.446 | 0.235 | 0.750 |
| 1 | occurrence | all | clim | p >= 0.3 | 0.404 | 0.516 | 0.453 | 0.135 | 0.812 |
| 1 | occurrence | all | clim | p >= 0.5 | 0.504 | 0.307 | 0.382 | 0.054 | 0.850 |
| 1 | occurrence | all | clim | p >= 0.7 | 0.600 | 0.125 | 0.207 | 0.015 | 0.855 |
| 1 | occurrence | all | pers | top 5% flagged | 0.335 | 0.234 | 0.276 | 0.083 | 0.814 |
| 1 | occurrence | all | pers | top 10% flagged | 0.335 | 0.234 | 0.276 | 0.083 | 0.814 |
| 1 | occurrence | all | pers | top 20% flagged | 0.151 | 1.000 | 0.262 | 1.000 | 0.151 |
| 1 | occurrence | all | pers | top 30% flagged | 0.151 | 1.000 | 0.262 | 1.000 | 0.151 |
| 1 | occurrence | all | pers | p >= 0.1 | 0.335 | 0.234 | 0.276 | 0.083 | 0.814 |
| 1 | occurrence | all | pers | p >= 0.2 | 0.335 | 0.234 | 0.276 | 0.083 | 0.814 |
| 1 | occurrence | all | pers | p >= 0.3 | 0.416 | 0.013 | 0.025 | 0.003 | 0.848 |
| 1 | occurrence | all | pers | p >= 0.5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.849 |
| 1 | occurrence | all | pers | p >= 0.7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.849 |
| 1 | occurrence | all | M1: gbm | top 5% flagged | 0.286 | 0.095 | 0.143 | 0.042 | 0.828 |
| 1 | occurrence | all | M1: gbm | top 10% flagged | 0.277 | 0.184 | 0.221 | 0.085 | 0.804 |
| 1 | occurrence | all | M1: gbm | top 20% flagged | 0.259 | 0.343 | 0.295 | 0.175 | 0.753 |
| 1 | occurrence | all | M1: gbm | top 30% flagged | 0.237 | 0.471 | 0.315 | 0.270 | 0.691 |
| 1 | occurrence | all | M1: gbm | p >= 0.1 | 0.215 | 0.658 | 0.324 | 0.428 | 0.585 |
| 1 | occurrence | all | M1: gbm | p >= 0.2 | 0.259 | 0.356 | 0.299 | 0.181 | 0.749 |
| 1 | occurrence | all | M1: gbm | p >= 0.3 | 0.281 | 0.152 | 0.197 | 0.069 | 0.813 |
| 1 | occurrence | all | M1: gbm | p >= 0.5 | 0.394 | 0.019 | 0.037 | 0.005 | 0.847 |
| 1 | occurrence | all | M1: gbm | p >= 0.7 | 0.600 | 0.001 | 0.002 | 0.000 | 0.849 |
| 1 | occurrence | all | M3: gbm | top 5% flagged | 0.614 | 0.203 | 0.305 | 0.023 | 0.860 |
| 1 | occurrence | all | M3: gbm | top 10% flagged | 0.513 | 0.340 | 0.409 | 0.057 | 0.852 |
| 1 | occurrence | all | M3: gbm | top 20% flagged | 0.402 | 0.533 | 0.458 | 0.141 | 0.810 |
| 1 | occurrence | all | M3: gbm | top 30% flagged | 0.333 | 0.661 | 0.443 | 0.236 | 0.749 |
| 1 | occurrence | all | M3: gbm | p >= 0.1 | 0.300 | 0.726 | 0.425 | 0.301 | 0.703 |
| 1 | occurrence | all | M3: gbm | p >= 0.2 | 0.371 | 0.602 | 0.459 | 0.181 | 0.786 |
| 1 | occurrence | all | M3: gbm | p >= 0.3 | 0.420 | 0.501 | 0.457 | 0.123 | 0.820 |
| 1 | occurrence | all | M3: gbm | p >= 0.5 | 0.520 | 0.328 | 0.402 | 0.054 | 0.853 |
| 1 | occurrence | all | M3: gbm | p >= 0.7 | 0.640 | 0.181 | 0.282 | 0.018 | 0.861 |
| 1 | occurrence | cropland_harvest | clim | top 5% flagged | 0.632 | 0.160 | 0.255 | 0.026 | 0.795 |
| 1 | occurrence | cropland_harvest | clim | top 10% flagged | 0.532 | 0.279 | 0.366 | 0.069 | 0.787 |
| 1 | occurrence | cropland_harvest | clim | top 20% flagged | 0.454 | 0.454 | 0.454 | 0.154 | 0.760 |
| 1 | occurrence | cropland_harvest | clim | top 30% flagged | 0.405 | 0.607 | 0.485 | 0.252 | 0.717 |
| 1 | occurrence | cropland_harvest | clim | p >= 0.1 | 0.319 | 0.837 | 0.462 | 0.505 | 0.570 |
| 1 | occurrence | cropland_harvest | clim | p >= 0.2 | 0.374 | 0.660 | 0.477 | 0.312 | 0.682 |
| 1 | occurrence | cropland_harvest | clim | p >= 0.3 | 0.445 | 0.514 | 0.477 | 0.181 | 0.752 |
| 1 | occurrence | cropland_harvest | clim | p >= 0.5 | 0.532 | 0.279 | 0.366 | 0.069 | 0.787 |
| 1 | occurrence | cropland_harvest | clim | p >= 0.7 | 0.617 | 0.096 | 0.166 | 0.017 | 0.788 |
| 1 | occurrence | cropland_harvest | pers | top 5% flagged | 0.313 | 0.237 | 0.270 | 0.146 | 0.718 |
| 1 | occurrence | cropland_harvest | pers | top 10% flagged | 0.313 | 0.237 | 0.270 | 0.146 | 0.718 |
| 1 | occurrence | cropland_harvest | pers | top 20% flagged | 0.220 | 1.000 | 0.361 | 1.000 | 0.220 |
| 1 | occurrence | cropland_harvest | pers | top 30% flagged | 0.220 | 1.000 | 0.361 | 1.000 | 0.220 |
| 1 | occurrence | cropland_harvest | pers | p >= 0.1 | 0.313 | 0.237 | 0.270 | 0.146 | 0.718 |
| 1 | occurrence | cropland_harvest | pers | p >= 0.2 | 0.313 | 0.237 | 0.270 | 0.146 | 0.718 |
| 1 | occurrence | cropland_harvest | pers | p >= 0.3 | 0.435 | 0.015 | 0.028 | 0.005 | 0.779 |
| 1 | occurrence | cropland_harvest | pers | p >= 0.5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.780 |
| 1 | occurrence | cropland_harvest | pers | p >= 0.7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.780 |
| 1 | occurrence | cropland_harvest | M1: gbm | top 5% flagged | 0.185 | 0.042 | 0.069 | 0.052 | 0.748 |
| 1 | occurrence | cropland_harvest | M1: gbm | top 10% flagged | 0.210 | 0.096 | 0.132 | 0.102 | 0.722 |
| 1 | occurrence | cropland_harvest | M1: gbm | top 20% flagged | 0.231 | 0.210 | 0.220 | 0.197 | 0.672 |
| 1 | occurrence | cropland_harvest | M1: gbm | top 30% flagged | 0.230 | 0.313 | 0.265 | 0.296 | 0.618 |
| 1 | occurrence | cropland_harvest | M1: gbm | p >= 0.1 | 0.245 | 0.653 | 0.357 | 0.567 | 0.481 |
| 1 | occurrence | cropland_harvest | M1: gbm | p >= 0.2 | 0.241 | 0.280 | 0.259 | 0.249 | 0.648 |
| 1 | occurrence | cropland_harvest | M1: gbm | p >= 0.3 | 0.203 | 0.087 | 0.122 | 0.096 | 0.724 |
| 1 | occurrence | cropland_harvest | M1: gbm | p >= 0.5 | 0.211 | 0.006 | 0.011 | 0.006 | 0.776 |
| 1 | occurrence | cropland_harvest | M1: gbm | p >= 0.7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.780 |
| 1 | occurrence | cropland_harvest | M3: gbm | top 5% flagged | 0.605 | 0.138 | 0.225 | 0.025 | 0.790 |
| 1 | occurrence | cropland_harvest | M3: gbm | top 10% flagged | 0.510 | 0.232 | 0.319 | 0.063 | 0.782 |
| 1 | occurrence | cropland_harvest | M3: gbm | top 20% flagged | 0.427 | 0.389 | 0.407 | 0.147 | 0.751 |
| 1 | occurrence | cropland_harvest | M3: gbm | top 30% flagged | 0.389 | 0.531 | 0.449 | 0.235 | 0.714 |
| 1 | occurrence | cropland_harvest | M3: gbm | p >= 0.1 | 0.340 | 0.752 | 0.468 | 0.412 | 0.624 |
| 1 | occurrence | cropland_harvest | M3: gbm | p >= 0.2 | 0.383 | 0.617 | 0.472 | 0.281 | 0.697 |
| 1 | occurrence | cropland_harvest | M3: gbm | p >= 0.3 | 0.403 | 0.502 | 0.447 | 0.210 | 0.727 |
| 1 | occurrence | cropland_harvest | M3: gbm | p >= 0.5 | 0.466 | 0.325 | 0.383 | 0.105 | 0.769 |
| 1 | occurrence | cropland_harvest | M3: gbm | p >= 0.7 | 0.579 | 0.160 | 0.250 | 0.033 | 0.790 |
| 1 | persistence | all | clim | top 5% flagged | 0.946 | 0.078 | 0.144 | 0.007 | 0.428 |
| 1 | persistence | all | clim | top 10% flagged | 0.934 | 0.189 | 0.315 | 0.022 | 0.491 |
| 1 | persistence | all | clim | top 20% flagged | 0.909 | 0.313 | 0.466 | 0.050 | 0.557 |
| 1 | persistence | all | clim | top 30% flagged | 0.883 | 0.466 | 0.610 | 0.100 | 0.632 |
| 1 | persistence | all | clim | p >= 0.1 | 0.696 | 0.968 | 0.810 | 0.682 | 0.720 |
| 1 | persistence | all | clim | p >= 0.2 | 0.740 | 0.919 | 0.820 | 0.522 | 0.751 |
| 1 | persistence | all | clim | p >= 0.3 | 0.776 | 0.838 | 0.806 | 0.390 | 0.751 |
| 1 | persistence | all | clim | p >= 0.5 | 0.837 | 0.654 | 0.735 | 0.205 | 0.708 |
| 1 | persistence | all | clim | p >= 0.7 | 0.901 | 0.376 | 0.530 | 0.067 | 0.589 |
| 1 | persistence | all | pers | top 5% flagged | 0.696 | 0.575 | 0.630 | 0.405 | 0.583 |
| 1 | persistence | all | pers | top 10% flagged | 0.696 | 0.575 | 0.630 | 0.405 | 0.583 |
| 1 | persistence | all | pers | top 20% flagged | 0.696 | 0.575 | 0.630 | 0.405 | 0.583 |
| 1 | persistence | all | pers | top 30% flagged | 0.696 | 0.575 | 0.630 | 0.405 | 0.583 |
| 1 | persistence | all | pers | p >= 0.1 | 0.617 | 1.000 | 0.763 | 1.000 | 0.617 |
| 1 | persistence | all | pers | p >= 0.2 | 0.617 | 1.000 | 0.763 | 1.000 | 0.617 |
| 1 | persistence | all | pers | p >= 0.3 | 0.617 | 1.000 | 0.763 | 1.000 | 0.617 |
| 1 | persistence | all | pers | p >= 0.5 | 0.617 | 1.000 | 0.763 | 1.000 | 0.617 |
| 1 | persistence | all | pers | p >= 0.7 | 0.669 | 0.747 | 0.706 | 0.597 | 0.615 |
| 1 | persistence | all | M1: gbm | top 5% flagged | 0.813 | 0.066 | 0.122 | 0.024 | 0.414 |
| 1 | persistence | all | M1: gbm | top 10% flagged | 0.786 | 0.127 | 0.219 | 0.056 | 0.440 |
| 1 | persistence | all | M1: gbm | top 20% flagged | 0.750 | 0.243 | 0.367 | 0.131 | 0.483 |
| 1 | persistence | all | M1: gbm | top 30% flagged | 0.726 | 0.353 | 0.475 | 0.215 | 0.518 |
| 1 | persistence | all | M1: gbm | p >= 0.1 | 0.619 | 0.998 | 0.764 | 0.991 | 0.619 |
| 1 | persistence | all | M1: gbm | p >= 0.2 | 0.633 | 0.960 | 0.763 | 0.900 | 0.631 |
| 1 | persistence | all | M1: gbm | p >= 0.3 | 0.653 | 0.885 | 0.751 | 0.760 | 0.638 |
| 1 | persistence | all | M1: gbm | p >= 0.5 | 0.689 | 0.613 | 0.649 | 0.446 | 0.590 |
| 1 | persistence | all | M1: gbm | p >= 0.7 | 0.750 | 0.256 | 0.382 | 0.138 | 0.488 |
| 1 | persistence | all | M3: gbm | top 5% flagged | 0.952 | 0.077 | 0.143 | 0.006 | 0.428 |
| 1 | persistence | all | M3: gbm | top 10% flagged | 0.937 | 0.152 | 0.261 | 0.016 | 0.470 |
| 1 | persistence | all | M3: gbm | top 20% flagged | 0.909 | 0.295 | 0.445 | 0.047 | 0.546 |
| 1 | persistence | all | M3: gbm | top 30% flagged | 0.885 | 0.430 | 0.579 | 0.090 | 0.614 |
| 1 | persistence | all | M3: gbm | p >= 0.1 | 0.693 | 0.971 | 0.809 | 0.693 | 0.717 |
| 1 | persistence | all | M3: gbm | p >= 0.2 | 0.725 | 0.941 | 0.819 | 0.576 | 0.743 |
| 1 | persistence | all | M3: gbm | p >= 0.3 | 0.750 | 0.903 | 0.820 | 0.485 | 0.754 |
| 1 | persistence | all | M3: gbm | p >= 0.5 | 0.796 | 0.795 | 0.795 | 0.330 | 0.747 |
| 1 | persistence | all | M3: gbm | p >= 0.7 | 0.848 | 0.625 | 0.720 | 0.181 | 0.699 |
| 1 | persistence | cropland_harvest | clim | top 5% flagged | 0.967 | 0.129 | 0.228 | 0.007 | 0.463 |
| 1 | persistence | cropland_harvest | clim | top 10% flagged | 0.969 | 0.175 | 0.296 | 0.009 | 0.491 |
| 1 | persistence | cropland_harvest | clim | top 20% flagged | 0.936 | 0.338 | 0.497 | 0.037 | 0.580 |
| 1 | persistence | cropland_harvest | clim | top 30% flagged | 0.906 | 0.513 | 0.655 | 0.084 | 0.669 |
| 1 | persistence | cropland_harvest | clim | p >= 0.1 | 0.708 | 0.961 | 0.815 | 0.630 | 0.733 |
| 1 | persistence | cropland_harvest | clim | p >= 0.2 | 0.760 | 0.919 | 0.832 | 0.460 | 0.773 |
| 1 | persistence | cropland_harvest | clim | p >= 0.3 | 0.800 | 0.852 | 0.825 | 0.339 | 0.778 |
| 1 | persistence | cropland_harvest | clim | p >= 0.5 | 0.863 | 0.688 | 0.766 | 0.174 | 0.742 |
| 1 | persistence | cropland_harvest | clim | p >= 0.7 | 0.926 | 0.400 | 0.559 | 0.051 | 0.613 |
| 1 | persistence | cropland_harvest | pers | top 5% flagged | 0.689 | 0.458 | 0.550 | 0.328 | 0.541 |
| 1 | persistence | cropland_harvest | pers | top 10% flagged | 0.689 | 0.458 | 0.550 | 0.328 | 0.541 |
| 1 | persistence | cropland_harvest | pers | top 20% flagged | 0.689 | 0.458 | 0.550 | 0.328 | 0.541 |
| 1 | persistence | cropland_harvest | pers | top 30% flagged | 0.689 | 0.458 | 0.550 | 0.328 | 0.541 |
| 1 | persistence | cropland_harvest | pers | p >= 0.1 | 0.613 | 1.000 | 0.760 | 1.000 | 0.613 |
| 1 | persistence | cropland_harvest | pers | p >= 0.2 | 0.613 | 1.000 | 0.760 | 1.000 | 0.613 |
| 1 | persistence | cropland_harvest | pers | p >= 0.3 | 0.613 | 1.000 | 0.760 | 1.000 | 0.613 |
| 1 | persistence | cropland_harvest | pers | p >= 0.5 | 0.613 | 1.000 | 0.760 | 1.000 | 0.613 |
| 1 | persistence | cropland_harvest | pers | p >= 0.7 | 0.661 | 0.671 | 0.666 | 0.547 | 0.587 |
| 1 | persistence | cropland_harvest | M1: gbm | top 5% flagged | 0.851 | 0.070 | 0.129 | 0.019 | 0.422 |
| 1 | persistence | cropland_harvest | M1: gbm | top 10% flagged | 0.838 | 0.137 | 0.236 | 0.042 | 0.455 |
| 1 | persistence | cropland_harvest | M1: gbm | top 20% flagged | 0.820 | 0.268 | 0.404 | 0.093 | 0.515 |
| 1 | persistence | cropland_harvest | M1: gbm | top 30% flagged | 0.735 | 0.360 | 0.483 | 0.205 | 0.528 |
| 1 | persistence | cropland_harvest | M1: gbm | p >= 0.1 | 0.613 | 0.999 | 0.760 | 0.998 | 0.613 |
| 1 | persistence | cropland_harvest | M1: gbm | p >= 0.2 | 0.620 | 0.957 | 0.752 | 0.932 | 0.613 |
| 1 | persistence | cropland_harvest | M1: gbm | p >= 0.3 | 0.640 | 0.877 | 0.740 | 0.784 | 0.621 |
| 1 | persistence | cropland_harvest | M1: gbm | p >= 0.5 | 0.689 | 0.597 | 0.640 | 0.428 | 0.588 |
| 1 | persistence | cropland_harvest | M1: gbm | p >= 0.7 | 0.821 | 0.264 | 0.400 | 0.091 | 0.514 |
| 1 | persistence | cropland_harvest | M3: gbm | top 5% flagged | 0.973 | 0.080 | 0.147 | 0.004 | 0.434 |
| 1 | persistence | cropland_harvest | M3: gbm | top 10% flagged | 0.973 | 0.159 | 0.274 | 0.007 | 0.482 |
| 1 | persistence | cropland_harvest | M3: gbm | top 20% flagged | 0.942 | 0.308 | 0.464 | 0.030 | 0.564 |
| 1 | persistence | cropland_harvest | M3: gbm | top 30% flagged | 0.905 | 0.442 | 0.594 | 0.074 | 0.630 |
| 1 | persistence | cropland_harvest | M3: gbm | p >= 0.1 | 0.700 | 0.957 | 0.809 | 0.649 | 0.723 |
| 1 | persistence | cropland_harvest | M3: gbm | p >= 0.2 | 0.749 | 0.923 | 0.827 | 0.489 | 0.763 |
| 1 | persistence | cropland_harvest | M3: gbm | p >= 0.3 | 0.779 | 0.895 | 0.833 | 0.404 | 0.780 |
| 1 | persistence | cropland_harvest | M3: gbm | p >= 0.5 | 0.823 | 0.812 | 0.817 | 0.277 | 0.777 |
| 1 | persistence | cropland_harvest | M3: gbm | p >= 0.7 | 0.860 | 0.664 | 0.749 | 0.172 | 0.727 |
| 1 | occurrence | all | M1_physical: gbm | top 5% flagged | 0.312 | 0.103 | 0.155 | 0.041 | 0.830 |
| 1 | occurrence | all | M1_physical: gbm | top 10% flagged | 0.293 | 0.194 | 0.234 | 0.083 | 0.808 |
| 1 | occurrence | all | M1_physical: gbm | top 20% flagged | 0.272 | 0.360 | 0.309 | 0.172 | 0.758 |
| 1 | occurrence | all | M1_physical: gbm | top 30% flagged | 0.248 | 0.494 | 0.330 | 0.266 | 0.698 |
| 1 | occurrence | all | M1_physical: gbm | p >= 0.1 | 0.219 | 0.716 | 0.335 | 0.454 | 0.571 |
| 1 | occurrence | all | M1_physical: gbm | p >= 0.2 | 0.266 | 0.385 | 0.314 | 0.189 | 0.746 |
| 1 | occurrence | all | M1_physical: gbm | p >= 0.3 | 0.298 | 0.172 | 0.218 | 0.072 | 0.814 |
| 1 | occurrence | all | M1_physical: gbm | p >= 0.5 | 0.350 | 0.020 | 0.039 | 0.007 | 0.846 |
| 1 | occurrence | all | M1_physical: gbm | p >= 0.7 | 0.545 | 0.001 | 0.002 | 0.000 | 0.849 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | top 5% flagged | 0.261 | 0.060 | 0.097 | 0.048 | 0.756 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | top 10% flagged | 0.242 | 0.110 | 0.152 | 0.097 | 0.728 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | top 20% flagged | 0.270 | 0.245 | 0.257 | 0.188 | 0.688 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | top 30% flagged | 0.284 | 0.388 | 0.328 | 0.276 | 0.650 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | p >= 0.1 | 0.240 | 0.779 | 0.367 | 0.696 | 0.408 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | p >= 0.2 | 0.275 | 0.464 | 0.346 | 0.345 | 0.613 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | p >= 0.3 | 0.251 | 0.176 | 0.206 | 0.148 | 0.703 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | p >= 0.5 | 0.278 | 0.015 | 0.028 | 0.011 | 0.775 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | p >= 0.7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.780 |
| 1 | persistence | all | M1_physical: gbm | top 5% flagged | 0.799 | 0.065 | 0.120 | 0.026 | 0.413 |
| 1 | persistence | all | M1_physical: gbm | top 10% flagged | 0.774 | 0.125 | 0.216 | 0.059 | 0.438 |
| 1 | persistence | all | M1_physical: gbm | top 20% flagged | 0.744 | 0.241 | 0.364 | 0.134 | 0.480 |
| 1 | persistence | all | M1_physical: gbm | top 30% flagged | 0.723 | 0.352 | 0.473 | 0.217 | 0.517 |
| 1 | persistence | all | M1_physical: gbm | p >= 0.1 | 0.618 | 0.999 | 0.763 | 0.998 | 0.618 |
| 1 | persistence | all | M1_physical: gbm | p >= 0.2 | 0.624 | 0.976 | 0.761 | 0.948 | 0.622 |
| 1 | persistence | all | M1_physical: gbm | p >= 0.3 | 0.642 | 0.897 | 0.748 | 0.808 | 0.627 |
| 1 | persistence | all | M1_physical: gbm | p >= 0.5 | 0.691 | 0.579 | 0.630 | 0.417 | 0.581 |
| 1 | persistence | all | M1_physical: gbm | p >= 0.7 | 0.749 | 0.202 | 0.319 | 0.109 | 0.466 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | top 5% flagged | 0.905 | 0.074 | 0.137 | 0.012 | 0.427 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | top 10% flagged | 0.858 | 0.140 | 0.241 | 0.037 | 0.459 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | top 20% flagged | 0.773 | 0.252 | 0.380 | 0.118 | 0.496 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | top 30% flagged | 0.717 | 0.351 | 0.471 | 0.219 | 0.517 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | p >= 0.1 | 0.614 | 1.000 | 0.761 | 0.996 | 0.615 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | p >= 0.2 | 0.624 | 0.976 | 0.761 | 0.933 | 0.624 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | p >= 0.3 | 0.639 | 0.941 | 0.762 | 0.842 | 0.638 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | p >= 0.5 | 0.671 | 0.660 | 0.666 | 0.514 | 0.593 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | p >= 0.7 | 0.773 | 0.260 | 0.389 | 0.121 | 0.499 |

## Ablation (strict), subset all

| target | h | direction | group | gbm_delta_pr_auc | gbm_noise_sd | gbm_years_positive | gbm_informative | logreg_delta_pr_auc | logreg_noise_sd | logreg_years_positive | logreg_informative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| occurrence | 1 | add | atmosphere | 0.005 | 0.001 | 4/7 | False | 0.040 | 0.000 | 6/7 | False |
| occurrence | 1 | add | climate_indices | -0.031 | 0.001 | 2/7 | False | -0.047 | 0.000 | 4/7 | False |
| occurrence | 1 | add | dartmouth | 0.015 | 0.001 | 6/7 | True | -0.052 | 0.000 | 2/7 | False |
| occurrence | 1 | add | evapotranspiration | -0.012 | 0.001 | 2/7 | False | 0.029 | 0.000 | 7/7 | False |
| occurrence | 1 | add | glofas | -0.000 | 0.001 | 4/7 | False | 0.019 | 0.000 | 5/7 | False |
| occurrence | 1 | add | heterogeneity | 0.005 | 0.001 | 4/7 | False | 0.001 | 0.000 | 5/7 | False |
| occurrence | 1 | add | lakes | -0.027 | 0.001 | 0/7 | False | -0.007 | 0.000 | 6/7 | False |
| occurrence | 1 | add | local_rain | 0.022 | 0.001 | 5/7 | False | 0.034 | 0.000 | 5/7 | False |
| occurrence | 1 | add | rain_shape | 0.005 | 0.001 | 3/7 | False | 0.029 | 0.000 | 5/7 | False |
| occurrence | 1 | add | soil | -0.005 | 0.001 | 1/7 | False | -0.002 | 0.000 | 2/7 | False |
| occurrence | 1 | add | soil_moisture | 0.014 | 0.001 | 4/7 | False | 0.033 | 0.000 | 5/7 | False |
| occurrence | 1 | add | terrain | -0.002 | 0.001 | 2/7 | False | 0.003 | 0.000 | 5/7 | False |
| occurrence | 1 | add | upstream_catchment_rain | 0.009 | 0.001 | 4/7 | False | 0.005 | 0.000 | 5/7 | False |
| occurrence | 1 | add | upstream_rain | 0.008 | 0.001 | 5/7 | False | 0.014 | 0.000 | 6/7 | False |
| occurrence | 1 | add | water_storage | -0.018 | 0.001 | 3/7 | False | 0.018 | 0.000 | 6/7 | False |
| occurrence | 1 | drop | atmosphere | -0.002 | 0.005 | 5/7 | False | 0.003 | 0.003 | 4/7 | False |
| occurrence | 1 | drop | climate_indices | 0.004 | 0.005 | 5/7 | False | -0.006 | 0.003 | 2/7 | False |
| occurrence | 1 | drop | dartmouth | 0.000 | 0.005 | 2/7 | False | -0.007 | 0.003 | 4/7 | False |
| occurrence | 1 | drop | evapotranspiration | 0.002 | 0.005 | 6/7 | False | 0.000 | 0.003 | 2/7 | False |
| occurrence | 1 | drop | glofas | 0.004 | 0.005 | 5/7 | False | -0.011 | 0.003 | 2/7 | False |
| occurrence | 1 | drop | heterogeneity | -0.002 | 0.005 | 2/7 | False | 0.000 | 0.003 | 7/7 | False |
| occurrence | 1 | drop | lakes | -0.000 | 0.005 | 4/7 | False | 0.001 | 0.003 | 4/7 | False |
| occurrence | 1 | drop | local_rain | -0.000 | 0.005 | 4/7 | False | -0.002 | 0.003 | 3/7 | False |
| occurrence | 1 | drop | rain_shape | 0.000 | 0.005 | 3/7 | False | -0.000 | 0.003 | 3/7 | False |
| occurrence | 1 | drop | soil | 0.004 | 0.005 | 4/7 | False | -0.001 | 0.003 | 3/7 | False |
| occurrence | 1 | drop | soil_moisture | -0.003 | 0.005 | 3/7 | False | 0.000 | 0.003 | 3/7 | False |
| occurrence | 1 | drop | terrain | 0.007 | 0.005 | 5/7 | False | 0.005 | 0.003 | 7/7 | False |
| occurrence | 1 | drop | upstream_catchment_rain | 0.002 | 0.005 | 4/7 | False | -0.001 | 0.003 | 1/7 | False |
| occurrence | 1 | drop | upstream_rain | 0.002 | 0.005 | 5/7 | False | -0.004 | 0.003 | 4/7 | False |
| occurrence | 1 | drop | water_storage | -0.010 | 0.005 | 2/7 | False | -0.009 | 0.003 | 3/7 | False |
| occurrence | 1 | fdrop | climate_indices | -0.002 | 0.004 | 3/7 | False |  |  |  |  |
| occurrence | 1 | fdrop | land | 0.044 | 0.004 | 7/7 | True |  |  |  |  |
| occurrence | 1 | fdrop | rivers_storage | 0.000 | 0.004 | 3/7 | False |  |  |  |  |
| occurrence | 1 | fdrop | weather | 0.081 | 0.004 | 7/7 | True |  |  |  |  |
| occurrence | 1 | pdrop | atmosphere | -0.004 | 0.004 | 2/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | climate_indices | -0.002 | 0.004 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | dartmouth | 0.008 | 0.004 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | evapotranspiration | -0.002 | 0.004 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | glofas | -0.011 | 0.004 | 2/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | heterogeneity | 0.006 | 0.004 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | lakes | 0.006 | 0.004 | 5/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | local_rain | 0.005 | 0.004 | 5/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | overbank | 0.003 | 0.004 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | rain_shape | 0.000 | 0.004 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | soil | 0.007 | 0.004 | 6/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | soil_moisture | -0.002 | 0.004 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | terrain | 0.005 | 0.004 | 5/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | upstream_catchment_rain | -0.004 | 0.004 | 2/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | upstream_rain | 0.007 | 0.004 | 6/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | water_storage | 0.001 | 0.004 | 3/7 | False |  |  |  |  |
| persistence | 1 | add | atmosphere | -0.000 | 0.001 | 2/7 | False | 0.013 | 0.000 | 6/7 | False |
| persistence | 1 | add | climate_indices | -0.010 | 0.001 | 3/7 | False | -0.000 | 0.000 | 3/7 | False |
| persistence | 1 | add | dartmouth | 0.004 | 0.001 | 5/7 | False | -0.021 | 0.000 | 1/7 | False |
| persistence | 1 | add | evapotranspiration | 0.004 | 0.001 | 3/7 | False | 0.004 | 0.000 | 2/7 | False |
| persistence | 1 | add | glofas | 0.007 | 0.001 | 2/7 | False | 0.017 | 0.000 | 5/7 | False |
| persistence | 1 | add | heterogeneity | 0.001 | 0.001 | 5/7 | False | 0.001 | 0.000 | 4/7 | False |
| persistence | 1 | add | lakes | -0.014 | 0.001 | 1/7 | False | 0.004 | 0.000 | 5/7 | False |
| persistence | 1 | add | local_rain | 0.010 | 0.001 | 3/7 | False | 0.013 | 0.000 | 5/7 | False |
| persistence | 1 | add | rain_shape | 0.010 | 0.001 | 2/7 | False | 0.011 | 0.000 | 3/7 | False |
| persistence | 1 | add | soil | -0.008 | 0.001 | 1/7 | False | -0.001 | 0.000 | 5/7 | False |
| persistence | 1 | add | soil_moisture | 0.007 | 0.001 | 2/7 | False | 0.006 | 0.000 | 4/7 | False |
| persistence | 1 | add | terrain | -0.002 | 0.001 | 2/7 | False | 0.005 | 0.000 | 4/7 | False |
| persistence | 1 | add | upstream_catchment_rain | 0.005 | 0.001 | 2/7 | False | 0.006 | 0.000 | 5/7 | False |
| persistence | 1 | add | upstream_rain | 0.010 | 0.001 | 5/7 | False | 0.010 | 0.000 | 6/7 | False |
| persistence | 1 | add | water_storage | 0.005 | 0.001 | 3/7 | False | 0.014 | 0.000 | 5/7 | False |
| persistence | 1 | drop | atmosphere | 0.004 | 0.002 | 5/7 | False | 0.007 | 0.008 | 3/7 | False |
| persistence | 1 | drop | climate_indices | -0.001 | 0.002 | 5/7 | False | -0.016 | 0.008 | 2/7 | False |
| persistence | 1 | drop | dartmouth | -0.004 | 0.002 | 2/7 | False | -0.013 | 0.008 | 2/7 | False |
| persistence | 1 | drop | evapotranspiration | 0.003 | 0.002 | 4/7 | False | 0.001 | 0.008 | 6/7 | False |
| persistence | 1 | drop | glofas | -0.002 | 0.002 | 4/7 | False | 0.008 | 0.008 | 3/7 | False |
| persistence | 1 | drop | heterogeneity | 0.001 | 0.002 | 6/7 | False | 0.006 | 0.008 | 7/7 | False |
| persistence | 1 | drop | lakes | -0.003 | 0.002 | 3/7 | False | -0.003 | 0.008 | 5/7 | False |
| persistence | 1 | drop | local_rain | 0.001 | 0.002 | 4/7 | False | -0.010 | 0.008 | 3/7 | False |
| persistence | 1 | drop | rain_shape | -0.002 | 0.002 | 3/7 | False | -0.000 | 0.008 | 3/7 | False |
| persistence | 1 | drop | soil | 0.009 | 0.002 | 5/7 | False | 0.005 | 0.008 | 5/7 | False |
| persistence | 1 | drop | soil_moisture | 0.002 | 0.002 | 4/7 | False | 0.004 | 0.008 | 5/7 | False |
| persistence | 1 | drop | terrain | 0.014 | 0.002 | 5/7 | False | 0.017 | 0.008 | 6/7 | False |
| persistence | 1 | drop | upstream_catchment_rain | 0.000 | 0.002 | 3/7 | False | 0.001 | 0.008 | 2/7 | False |
| persistence | 1 | drop | upstream_rain | 0.000 | 0.002 | 3/7 | False | -0.019 | 0.008 | 3/7 | False |
| persistence | 1 | drop | water_storage | 0.002 | 0.002 | 5/7 | False | 0.010 | 0.008 | 5/7 | False |
| persistence | 1 | fdrop | climate_indices | -0.002 | 0.006 | 3/7 | False |  |  |  |  |
| persistence | 1 | fdrop | land | 0.067 | 0.006 | 7/7 | True |  |  |  |  |
| persistence | 1 | fdrop | rivers_storage | -0.006 | 0.006 | 3/7 | False |  |  |  |  |
| persistence | 1 | fdrop | weather | 0.012 | 0.006 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | atmosphere | 0.000 | 0.006 | 2/7 | False |  |  |  |  |
| persistence | 1 | pdrop | climate_indices | -0.002 | 0.006 | 3/7 | False |  |  |  |  |
| persistence | 1 | pdrop | dartmouth | 0.000 | 0.006 | 2/7 | False |  |  |  |  |
| persistence | 1 | pdrop | evapotranspiration | -0.001 | 0.006 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | glofas | -0.005 | 0.006 | 3/7 | False |  |  |  |  |
| persistence | 1 | pdrop | heterogeneity | 0.005 | 0.006 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | lakes | -0.000 | 0.006 | 3/7 | False |  |  |  |  |
| persistence | 1 | pdrop | local_rain | -0.000 | 0.006 | 1/7 | False |  |  |  |  |
| persistence | 1 | pdrop | overbank | -0.001 | 0.006 | 5/7 | False |  |  |  |  |
| persistence | 1 | pdrop | rain_shape | 0.002 | 0.006 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | soil | 0.012 | 0.006 | 5/7 | False |  |  |  |  |
| persistence | 1 | pdrop | soil_moisture | 0.003 | 0.006 | 5/7 | False |  |  |  |  |
| persistence | 1 | pdrop | terrain | 0.032 | 0.006 | 6/7 | True |  |  |  |  |
| persistence | 1 | pdrop | upstream_catchment_rain | 0.001 | 0.006 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | upstream_rain | 0.001 | 0.006 | 3/7 | False |  |  |  |  |
| persistence | 1 | pdrop | water_storage | -0.004 | 0.006 | 2/7 | False |  |  |  |  |

## Ablation (strict), subset cropland

| target | h | direction | group | gbm_delta_pr_auc | gbm_noise_sd | gbm_years_positive | gbm_informative | logreg_delta_pr_auc | logreg_noise_sd | logreg_years_positive | logreg_informative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| occurrence | 1 | add | atmosphere | 0.016 | 0.003 | 6/7 | False | 0.049 | 0.000 | 5/7 | False |
| occurrence | 1 | add | climate_indices | -0.018 | 0.003 | 3/7 | False | -0.054 | 0.000 | 2/7 | False |
| occurrence | 1 | add | dartmouth | 0.022 | 0.003 | 5/7 | False | -0.071 | 0.000 | 2/7 | False |
| occurrence | 1 | add | evapotranspiration | -0.006 | 0.003 | 2/7 | False | 0.012 | 0.000 | 4/7 | False |
| occurrence | 1 | add | glofas | -0.002 | 0.003 | 4/7 | False | -0.002 | 0.000 | 3/7 | False |
| occurrence | 1 | add | heterogeneity | 0.003 | 0.003 | 2/7 | False | -0.002 | 0.000 | 2/7 | False |
| occurrence | 1 | add | lakes | -0.025 | 0.003 | 1/7 | False | -0.018 | 0.000 | 5/7 | False |
| occurrence | 1 | add | local_rain | 0.024 | 0.003 | 4/7 | False | 0.031 | 0.000 | 5/7 | False |
| occurrence | 1 | add | rain_shape | 0.020 | 0.003 | 2/7 | False | 0.020 | 0.000 | 5/7 | False |
| occurrence | 1 | add | soil | -0.004 | 0.003 | 3/7 | False | -0.004 | 0.000 | 3/7 | False |
| occurrence | 1 | add | soil_moisture | 0.021 | 0.003 | 4/7 | False | 0.025 | 0.000 | 5/7 | False |
| occurrence | 1 | add | terrain | -0.004 | 0.003 | 2/7 | False | -0.000 | 0.000 | 4/7 | False |
| occurrence | 1 | add | upstream_catchment_rain | 0.021 | 0.003 | 4/7 | False | 0.009 | 0.000 | 5/7 | False |
| occurrence | 1 | add | upstream_rain | 0.001 | 0.003 | 5/7 | False | 0.000 | 0.000 | 4/7 | False |
| occurrence | 1 | add | water_storage | 0.005 | 0.003 | 2/7 | False | 0.021 | 0.000 | 5/7 | False |
| occurrence | 1 | drop | atmosphere | -0.015 | 0.013 | 2/7 | False | 0.004 | 0.001 | 4/7 | False |
| occurrence | 1 | drop | climate_indices | 0.001 | 0.013 | 4/7 | False | -0.006 | 0.001 | 3/7 | False |
| occurrence | 1 | drop | dartmouth | -0.010 | 0.013 | 1/7 | False | -0.025 | 0.001 | 3/7 | False |
| occurrence | 1 | drop | evapotranspiration | 0.000 | 0.013 | 5/7 | False | 0.000 | 0.001 | 3/7 | False |
| occurrence | 1 | drop | glofas | -0.006 | 0.013 | 4/7 | False | -0.003 | 0.001 | 3/7 | False |
| occurrence | 1 | drop | heterogeneity | -0.010 | 0.013 | 4/7 | False | 0.000 | 0.001 | 4/7 | False |
| occurrence | 1 | drop | lakes | -0.005 | 0.013 | 5/7 | False | 0.002 | 0.001 | 5/7 | False |
| occurrence | 1 | drop | local_rain | -0.015 | 0.013 | 2/7 | False | -0.002 | 0.001 | 2/7 | False |
| occurrence | 1 | drop | rain_shape | -0.006 | 0.013 | 3/7 | False | 0.001 | 0.001 | 4/7 | False |
| occurrence | 1 | drop | soil | -0.009 | 0.013 | 5/7 | False | -0.002 | 0.001 | 0/7 | False |
| occurrence | 1 | drop | soil_moisture | -0.011 | 0.013 | 3/7 | False | -0.001 | 0.001 | 2/7 | False |
| occurrence | 1 | drop | terrain | 0.003 | 0.013 | 4/7 | False | 0.005 | 0.001 | 6/7 | False |
| occurrence | 1 | drop | upstream_catchment_rain | 0.000 | 0.013 | 4/7 | False | -0.001 | 0.001 | 3/7 | False |
| occurrence | 1 | drop | upstream_rain | -0.002 | 0.013 | 5/7 | False | -0.001 | 0.001 | 5/7 | False |
| occurrence | 1 | drop | water_storage | -0.001 | 0.013 | 3/7 | False | -0.003 | 0.001 | 3/7 | False |
| occurrence | 1 | fdrop | climate_indices | -0.002 | 0.009 | 1/7 | False |  |  |  |  |
| occurrence | 1 | fdrop | land | 0.035 | 0.009 | 5/7 | False |  |  |  |  |
| occurrence | 1 | fdrop | rivers_storage | -0.001 | 0.009 | 4/7 | False |  |  |  |  |
| occurrence | 1 | fdrop | weather | 0.068 | 0.009 | 5/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | atmosphere | -0.010 | 0.009 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | climate_indices | -0.002 | 0.009 | 1/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | dartmouth | 0.009 | 0.009 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | evapotranspiration | 0.003 | 0.009 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | glofas | -0.013 | 0.009 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | heterogeneity | 0.008 | 0.009 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | lakes | 0.009 | 0.009 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | local_rain | 0.018 | 0.009 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | overbank | 0.012 | 0.009 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | rain_shape | 0.012 | 0.009 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | soil | 0.007 | 0.009 | 2/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | soil_moisture | 0.004 | 0.009 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | terrain | 0.027 | 0.009 | 5/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | upstream_catchment_rain | -0.005 | 0.009 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | upstream_rain | 0.015 | 0.009 | 6/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | water_storage | 0.001 | 0.009 | 3/7 | False |  |  |  |  |
| persistence | 1 | add | atmosphere | -0.008 | 0.001 | 2/7 | False | 0.013 | 0.000 | 5/7 | False |
| persistence | 1 | add | climate_indices | -0.036 | 0.001 | 0/7 | False | -0.003 | 0.000 | 6/7 | False |
| persistence | 1 | add | dartmouth | -0.009 | 0.001 | 4/7 | False | -0.032 | 0.000 | 1/7 | False |
| persistence | 1 | add | evapotranspiration | -0.002 | 0.001 | 3/7 | False | 0.006 | 0.000 | 3/7 | False |
| persistence | 1 | add | glofas | -0.008 | 0.001 | 2/7 | False | 0.012 | 0.000 | 2/7 | False |
| persistence | 1 | add | heterogeneity | -0.004 | 0.001 | 4/7 | False | -0.000 | 0.000 | 2/7 | False |
| persistence | 1 | add | lakes | -0.017 | 0.001 | 2/7 | False | -0.002 | 0.000 | 6/7 | False |
| persistence | 1 | add | local_rain | 0.013 | 0.001 | 5/7 | False | 0.014 | 0.000 | 6/7 | False |
| persistence | 1 | add | rain_shape | 0.003 | 0.001 | 4/7 | False | 0.008 | 0.000 | 5/7 | False |
| persistence | 1 | add | soil | 0.008 | 0.001 | 5/7 | False | -0.001 | 0.000 | 4/7 | False |
| persistence | 1 | add | soil_moisture | 0.004 | 0.001 | 5/7 | False | 0.010 | 0.000 | 4/7 | False |
| persistence | 1 | add | terrain | -0.010 | 0.001 | 0/7 | False | 0.002 | 0.000 | 4/7 | False |
| persistence | 1 | add | upstream_catchment_rain | -0.008 | 0.001 | 3/7 | False | 0.006 | 0.000 | 5/7 | False |
| persistence | 1 | add | upstream_rain | -0.007 | 0.001 | 4/7 | False | -0.002 | 0.000 | 3/7 | False |
| persistence | 1 | add | water_storage | -0.010 | 0.001 | 2/7 | False | 0.010 | 0.000 | 5/7 | False |
| persistence | 1 | drop | atmosphere | 0.009 | 0.009 | 3/7 | False | -0.006 | 0.006 | 3/7 | False |
| persistence | 1 | drop | climate_indices | -0.008 | 0.009 | 4/7 | False | -0.008 | 0.006 | 4/7 | False |
| persistence | 1 | drop | dartmouth | -0.011 | 0.009 | 2/7 | False | -0.013 | 0.006 | 1/7 | False |
| persistence | 1 | drop | evapotranspiration | -0.003 | 0.009 | 2/7 | False | 0.003 | 0.006 | 5/7 | False |
| persistence | 1 | drop | glofas | 0.003 | 0.009 | 3/7 | False | 0.006 | 0.006 | 5/7 | False |
| persistence | 1 | drop | heterogeneity | -0.000 | 0.009 | 4/7 | False | 0.006 | 0.006 | 5/7 | False |
| persistence | 1 | drop | lakes | -0.005 | 0.009 | 3/7 | False | -0.001 | 0.006 | 6/7 | False |
| persistence | 1 | drop | local_rain | 0.003 | 0.009 | 3/7 | False | -0.008 | 0.006 | 3/7 | False |
| persistence | 1 | drop | rain_shape | -0.005 | 0.009 | 2/7 | False | -0.003 | 0.006 | 1/7 | False |
| persistence | 1 | drop | soil | 0.021 | 0.009 | 7/7 | False | 0.001 | 0.006 | 5/7 | False |
| persistence | 1 | drop | soil_moisture | -0.006 | 0.009 | 1/7 | False | 0.003 | 0.006 | 1/7 | False |
| persistence | 1 | drop | terrain | -0.009 | 0.009 | 5/7 | False | 0.016 | 0.006 | 5/7 | False |
| persistence | 1 | drop | upstream_catchment_rain | 0.002 | 0.009 | 5/7 | False | -0.004 | 0.006 | 3/7 | False |
| persistence | 1 | drop | upstream_rain | -0.005 | 0.009 | 4/7 | False | -0.014 | 0.006 | 5/7 | False |
| persistence | 1 | drop | water_storage | 0.002 | 0.009 | 3/7 | False | 0.004 | 0.006 | 4/7 | False |
| persistence | 1 | fdrop | climate_indices | -0.001 | 0.016 | 6/7 | False |  |  |  |  |
| persistence | 1 | fdrop | land | 0.075 | 0.016 | 6/7 | False |  |  |  |  |
| persistence | 1 | fdrop | rivers_storage | -0.007 | 0.016 | 5/7 | False |  |  |  |  |
| persistence | 1 | fdrop | weather | -0.001 | 0.016 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | atmosphere | 0.003 | 0.016 | 2/7 | False |  |  |  |  |
| persistence | 1 | pdrop | climate_indices | -0.001 | 0.016 | 6/7 | False |  |  |  |  |
| persistence | 1 | pdrop | dartmouth | 0.009 | 0.016 | 5/7 | False |  |  |  |  |
| persistence | 1 | pdrop | evapotranspiration | 0.002 | 0.016 | 3/7 | False |  |  |  |  |
| persistence | 1 | pdrop | glofas | -0.004 | 0.016 | 2/7 | False |  |  |  |  |
| persistence | 1 | pdrop | heterogeneity | 0.018 | 0.016 | 3/7 | False |  |  |  |  |
| persistence | 1 | pdrop | lakes | 0.008 | 0.016 | 3/7 | False |  |  |  |  |
| persistence | 1 | pdrop | local_rain | 0.009 | 0.016 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | overbank | 0.003 | 0.016 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | rain_shape | 0.007 | 0.016 | 3/7 | False |  |  |  |  |
| persistence | 1 | pdrop | soil | 0.028 | 0.016 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | soil_moisture | 0.016 | 0.016 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | terrain | 0.031 | 0.016 | 5/7 | False |  |  |  |  |
| persistence | 1 | pdrop | upstream_catchment_rain | 0.006 | 0.016 | 5/7 | False |  |  |  |  |
| persistence | 1 | pdrop | upstream_rain | 0.009 | 0.016 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | water_storage | -0.006 | 0.016 | 3/7 | False |  |  |  |  |

## Ablation (strict), subset cropland_harvest

| target | h | direction | group | gbm_delta_pr_auc | gbm_noise_sd | gbm_years_positive | gbm_informative | logreg_delta_pr_auc | logreg_noise_sd | logreg_years_positive | logreg_informative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| occurrence | 1 | add | atmosphere | -0.004 | 0.005 | 4/7 | False | 0.044 | 0.001 | 6/7 | False |
| occurrence | 1 | add | climate_indices | -0.004 | 0.005 | 4/7 | False | -0.072 | 0.001 | 3/7 | False |
| occurrence | 1 | add | dartmouth | 0.009 | 0.005 | 4/7 | False | -0.103 | 0.001 | 3/7 | False |
| occurrence | 1 | add | evapotranspiration | -0.012 | 0.005 | 3/7 | False | 0.009 | 0.001 | 5/7 | False |
| occurrence | 1 | add | glofas | -0.019 | 0.005 | 3/7 | False | -0.013 | 0.001 | 3/7 | False |
| occurrence | 1 | add | heterogeneity | 0.009 | 0.005 | 3/7 | False | -0.007 | 0.001 | 1/7 | False |
| occurrence | 1 | add | lakes | 0.000 | 0.005 | 4/7 | False | -0.039 | 0.001 | 4/7 | False |
| occurrence | 1 | add | local_rain | -0.005 | 0.005 | 4/7 | False | 0.013 | 0.001 | 4/7 | False |
| occurrence | 1 | add | rain_shape | 0.003 | 0.005 | 3/7 | False | -0.001 | 0.001 | 6/7 | False |
| occurrence | 1 | add | soil | 0.001 | 0.005 | 5/7 | False | 0.002 | 0.001 | 3/7 | False |
| occurrence | 1 | add | soil_moisture | 0.021 | 0.005 | 6/7 | False | 0.007 | 0.001 | 5/7 | False |
| occurrence | 1 | add | terrain | 0.011 | 0.005 | 4/7 | False | -0.003 | 0.001 | 2/7 | False |
| occurrence | 1 | add | upstream_catchment_rain | 0.000 | 0.005 | 3/7 | False | -0.003 | 0.001 | 3/7 | False |
| occurrence | 1 | add | upstream_rain | -0.013 | 0.005 | 4/7 | False | -0.027 | 0.001 | 4/7 | False |
| occurrence | 1 | add | water_storage | -0.026 | 0.005 | 2/7 | False | 0.007 | 0.001 | 4/7 | False |
| occurrence | 1 | drop | atmosphere | -0.021 | 0.020 | 2/7 | False | 0.010 | 0.002 | 3/7 | False |
| occurrence | 1 | drop | climate_indices | 0.004 | 0.020 | 5/7 | False | 0.001 | 0.002 | 4/7 | False |
| occurrence | 1 | drop | dartmouth | -0.014 | 0.020 | 1/7 | False | -0.034 | 0.002 | 0/7 | False |
| occurrence | 1 | drop | evapotranspiration | -0.005 | 0.020 | 2/7 | False | -0.001 | 0.002 | 1/7 | False |
| occurrence | 1 | drop | glofas | -0.009 | 0.020 | 6/7 | False | 0.000 | 0.002 | 6/7 | False |
| occurrence | 1 | drop | heterogeneity | -0.005 | 0.020 | 3/7 | False | -0.000 | 0.002 | 4/7 | False |
| occurrence | 1 | drop | lakes | -0.008 | 0.020 | 4/7 | False | 0.000 | 0.002 | 3/7 | False |
| occurrence | 1 | drop | local_rain | -0.018 | 0.020 | 3/7 | False | -0.001 | 0.002 | 2/7 | False |
| occurrence | 1 | drop | rain_shape | -0.012 | 0.020 | 4/7 | False | 0.001 | 0.002 | 6/7 | False |
| occurrence | 1 | drop | soil | -0.009 | 0.020 | 3/7 | False | 0.001 | 0.002 | 4/7 | False |
| occurrence | 1 | drop | soil_moisture | -0.019 | 0.020 | 3/7 | False | -0.001 | 0.002 | 2/7 | False |
| occurrence | 1 | drop | terrain | 0.003 | 0.020 | 4/7 | False | 0.004 | 0.002 | 5/7 | False |
| occurrence | 1 | drop | upstream_catchment_rain | -0.012 | 0.020 | 3/7 | False | 0.001 | 0.002 | 3/7 | False |
| occurrence | 1 | drop | upstream_rain | -0.003 | 0.020 | 5/7 | False | 0.001 | 0.002 | 3/7 | False |
| occurrence | 1 | drop | water_storage | 0.003 | 0.020 | 4/7 | False | -0.000 | 0.002 | 3/7 | False |
| occurrence | 1 | fdrop | climate_indices | -0.005 | 0.015 | 1/7 | False |  |  |  |  |
| occurrence | 1 | fdrop | land | 0.034 | 0.015 | 4/7 | False |  |  |  |  |
| occurrence | 1 | fdrop | rivers_storage | -0.028 | 0.015 | 2/7 | False |  |  |  |  |
| occurrence | 1 | fdrop | weather | 0.037 | 0.015 | 6/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | atmosphere | -0.029 | 0.015 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | climate_indices | -0.005 | 0.015 | 1/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | dartmouth | 0.007 | 0.015 | 2/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | evapotranspiration | -0.002 | 0.015 | 2/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | glofas | -0.031 | 0.015 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | heterogeneity | 0.005 | 0.015 | 1/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | lakes | 0.007 | 0.015 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | local_rain | 0.020 | 0.015 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | overbank | 0.014 | 0.015 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | rain_shape | 0.020 | 0.015 | 2/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | soil | 0.003 | 0.015 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | soil_moisture | -0.003 | 0.015 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | terrain | 0.033 | 0.015 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | upstream_catchment_rain | -0.009 | 0.015 | 3/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | upstream_rain | 0.001 | 0.015 | 4/7 | False |  |  |  |  |
| occurrence | 1 | pdrop | water_storage | 0.003 | 0.015 | 1/7 | False |  |  |  |  |
| persistence | 1 | add | atmosphere | -0.002 | 0.001 | 5/7 | False | 0.007 | 0.000 | 6/7 | False |
| persistence | 1 | add | climate_indices | -0.004 | 0.001 | 3/7 | False | -0.002 | 0.000 | 4/7 | False |
| persistence | 1 | add | dartmouth | -0.003 | 0.001 | 4/7 | False | -0.057 | 0.000 | 2/7 | False |
| persistence | 1 | add | evapotranspiration | 0.001 | 0.001 | 3/7 | False | -0.000 | 0.000 | 5/7 | False |
| persistence | 1 | add | glofas | -0.009 | 0.001 | 4/7 | False | 0.006 | 0.000 | 4/7 | False |
| persistence | 1 | add | heterogeneity | 0.005 | 0.001 | 7/7 | True | -0.001 | 0.000 | 2/7 | False |
| persistence | 1 | add | lakes | -0.002 | 0.001 | 3/7 | False | -0.004 | 0.000 | 3/7 | False |
| persistence | 1 | add | local_rain | 0.012 | 0.001 | 5/7 | False | 0.002 | 0.000 | 7/7 | False |
| persistence | 1 | add | rain_shape | -0.001 | 0.001 | 5/7 | False | -0.003 | 0.000 | 4/7 | False |
| persistence | 1 | add | soil | 0.010 | 0.001 | 6/7 | True | 0.000 | 0.000 | 3/7 | False |
| persistence | 1 | add | soil_moisture | 0.006 | 0.001 | 5/7 | False | -0.000 | 0.000 | 2/7 | False |
| persistence | 1 | add | terrain | -0.009 | 0.001 | 2/7 | False | -0.003 | 0.000 | 1/7 | False |
| persistence | 1 | add | upstream_catchment_rain | -0.010 | 0.001 | 3/7 | False | 0.004 | 0.000 | 4/7 | False |
| persistence | 1 | add | upstream_rain | -0.012 | 0.001 | 2/7 | False | -0.016 | 0.000 | 3/7 | False |
| persistence | 1 | add | water_storage | -0.016 | 0.001 | 3/7 | False | -0.000 | 0.000 | 4/7 | False |
| persistence | 1 | drop | atmosphere | 0.011 | 0.013 | 3/7 | False | 0.009 | 0.011 | 2/7 | False |
| persistence | 1 | drop | climate_indices | -0.004 | 0.013 | 1/7 | False | -0.009 | 0.011 | 2/7 | False |
| persistence | 1 | drop | dartmouth | -0.006 | 0.013 | 2/7 | False | -0.006 | 0.011 | 3/7 | False |
| persistence | 1 | drop | evapotranspiration | 0.001 | 0.013 | 1/7 | False | 0.011 | 0.011 | 5/7 | False |
| persistence | 1 | drop | glofas | -0.001 | 0.013 | 4/7 | False | 0.002 | 0.011 | 3/7 | False |
| persistence | 1 | drop | heterogeneity | 0.006 | 0.013 | 4/7 | False | 0.009 | 0.011 | 5/7 | False |
| persistence | 1 | drop | lakes | -0.000 | 0.013 | 3/7 | False | -0.005 | 0.011 | 3/7 | False |
| persistence | 1 | drop | local_rain | 0.010 | 0.013 | 4/7 | False | -0.004 | 0.011 | 5/7 | False |
| persistence | 1 | drop | rain_shape | -0.001 | 0.013 | 2/7 | False | 0.000 | 0.011 | 0/7 | False |
| persistence | 1 | drop | soil | 0.027 | 0.013 | 5/7 | False | 0.009 | 0.011 | 3/7 | False |
| persistence | 1 | drop | soil_moisture | -0.001 | 0.013 | 2/7 | False | -0.001 | 0.011 | 1/7 | False |
| persistence | 1 | drop | terrain | -0.007 | 0.013 | 4/7 | False | 0.006 | 0.011 | 5/7 | False |
| persistence | 1 | drop | upstream_catchment_rain | 0.015 | 0.013 | 4/7 | False | -0.003 | 0.011 | 4/7 | False |
| persistence | 1 | drop | upstream_rain | 0.007 | 0.013 | 4/7 | False | -0.016 | 0.011 | 3/7 | False |
| persistence | 1 | drop | water_storage | 0.001 | 0.013 | 3/7 | False | 0.008 | 0.011 | 4/7 | False |
| persistence | 1 | fdrop | climate_indices | 0.002 | 0.020 | 4/7 | False |  |  |  |  |
| persistence | 1 | fdrop | land | 0.075 | 0.020 | 5/7 | False |  |  |  |  |
| persistence | 1 | fdrop | rivers_storage | 0.022 | 0.020 | 4/7 | False |  |  |  |  |
| persistence | 1 | fdrop | weather | 0.009 | 0.020 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | atmosphere | 0.004 | 0.020 | 2/7 | False |  |  |  |  |
| persistence | 1 | pdrop | climate_indices | 0.002 | 0.020 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | dartmouth | 0.012 | 0.020 | 3/7 | False |  |  |  |  |
| persistence | 1 | pdrop | evapotranspiration | -0.000 | 0.020 | 2/7 | False |  |  |  |  |
| persistence | 1 | pdrop | glofas | -0.004 | 0.020 | 1/7 | False |  |  |  |  |
| persistence | 1 | pdrop | heterogeneity | 0.023 | 0.020 | 3/7 | False |  |  |  |  |
| persistence | 1 | pdrop | lakes | 0.002 | 0.020 | 2/7 | False |  |  |  |  |
| persistence | 1 | pdrop | local_rain | 0.006 | 0.020 | 2/7 | False |  |  |  |  |
| persistence | 1 | pdrop | overbank | 0.013 | 0.020 | 5/7 | False |  |  |  |  |
| persistence | 1 | pdrop | rain_shape | 0.012 | 0.020 | 2/7 | False |  |  |  |  |
| persistence | 1 | pdrop | soil | 0.039 | 0.020 | 6/7 | False |  |  |  |  |
| persistence | 1 | pdrop | soil_moisture | 0.014 | 0.020 | 2/7 | False |  |  |  |  |
| persistence | 1 | pdrop | terrain | 0.027 | 0.020 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | upstream_catchment_rain | 0.017 | 0.020 | 4/7 | False |  |  |  |  |
| persistence | 1 | pdrop | upstream_rain | 0.011 | 0.020 | 5/7 | False |  |  |  |  |
| persistence | 1 | pdrop | water_storage | 0.002 | 0.020 | 3/7 | False |  |  |  |  |

## Families of inputs removed from M1-physical (strict): PR-AUC loss, 95% interval over test years

| h | target | subset | family | pr_auc_loss_without_family | lo | hi |
|---|---|---|---|---|---|---|
| 1 | occurrence | all | weather | 0.081 | 0.053 | 0.099 |
| 1 | occurrence | all | rivers_storage | 0.000 | -0.009 | 0.010 |
| 1 | occurrence | all | land | 0.044 | 0.033 | 0.056 |
| 1 | occurrence | all | climate_indices | -0.002 | -0.011 | 0.007 |
| 1 | occurrence | cropland_harvest | weather | 0.037 | 0.005 | 0.087 |
| 1 | occurrence | cropland_harvest | rivers_storage | -0.028 | -0.049 | 0.012 |
| 1 | occurrence | cropland_harvest | land | 0.034 | 0.010 | 0.083 |
| 1 | occurrence | cropland_harvest | climate_indices | -0.005 | -0.042 | 0.019 |
| 1 | persistence | all | weather | 0.012 | -0.002 | 0.020 |
| 1 | persistence | all | rivers_storage | -0.006 | -0.013 | 0.009 |
| 1 | persistence | all | land | 0.067 | 0.043 | 0.088 |
| 1 | persistence | all | climate_indices | -0.002 | -0.005 | 0.002 |
| 1 | persistence | cropland_harvest | weather | 0.009 | -0.047 | 0.043 |
| 1 | persistence | cropland_harvest | rivers_storage | 0.022 | -0.004 | 0.039 |
| 1 | persistence | cropland_harvest | land | 0.075 | -0.009 | 0.115 |
| 1 | persistence | cropland_harvest | climate_indices | 0.002 | -0.015 | 0.010 |

## PR-AUC with 95% interval (bootstrap over test years, strict)

| h | target | subset | what | value | lo | hi |
|---|---|---|---|---|---|---|
| 1 | occurrence | all | clim | 0.410 | 0.353 | 0.474 |
| 1 | occurrence | all | pers | 0.195 | 0.181 | 0.206 |
| 1 | occurrence | all | M1: gbm | 0.242 | 0.206 | 0.278 |
| 1 | occurrence | all | M2: gbm | 0.411 | 0.381 | 0.446 |
| 1 | occurrence | all | M3: gbm | 0.443 | 0.399 | 0.490 |
| 1 | occurrence | all | M3: gbm - clim | 0.034 | 0.013 | 0.053 |
| 1 | occurrence | cropland_harvest | clim | 0.447 | 0.363 | 0.504 |
| 1 | occurrence | cropland_harvest | pers | 0.244 | 0.203 | 0.295 |
| 1 | occurrence | cropland_harvest | M1: gbm | 0.233 | 0.175 | 0.310 |
| 1 | occurrence | cropland_harvest | M2: gbm | 0.415 | 0.299 | 0.518 |
| 1 | occurrence | cropland_harvest | M3: gbm | 0.431 | 0.331 | 0.523 |
| 1 | occurrence | cropland_harvest | M3: gbm - clim | -0.016 | -0.059 | 0.044 |
| 1 | persistence | all | clim | 0.853 | 0.839 | 0.875 |
| 1 | persistence | all | pers | 0.672 | 0.650 | 0.705 |
| 1 | persistence | all | M1: gbm | 0.713 | 0.675 | 0.741 |
| 1 | persistence | all | M2: gbm | 0.836 | 0.822 | 0.855 |
| 1 | persistence | all | M3: gbm | 0.858 | 0.841 | 0.881 |
| 1 | persistence | all | M3: gbm - clim | 0.004 | 0.001 | 0.011 |
| 1 | persistence | cropland_harvest | clim | 0.882 | 0.846 | 0.911 |
| 1 | persistence | cropland_harvest | pers | 0.659 | 0.606 | 0.693 |
| 1 | persistence | cropland_harvest | M1: gbm | 0.735 | 0.682 | 0.765 |
| 1 | persistence | cropland_harvest | M2: gbm | 0.879 | 0.873 | 0.892 |
| 1 | persistence | cropland_harvest | M3: gbm | 0.883 | 0.850 | 0.904 |
| 1 | persistence | cropland_harvest | M3: gbm - clim | 0.001 | -0.010 | 0.011 |
| 1 | occurrence | all | M1_physical: gbm | 0.250 | 0.215 | 0.281 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | 0.256 | 0.189 | 0.339 |
| 1 | persistence | all | M1_physical: gbm | 0.708 | 0.672 | 0.735 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | 0.728 | 0.663 | 0.762 |

## Interactions and tree size (GBM, strict, h=1), PR-AUC

| target | subset | M1 | M1_additive | M1_large_trees | M1_small_trees |
|---|---|---|---|---|---|
| occurrence | all | 0.242 | 0.227 | 0.237 | 0.247 |
| occurrence | cropland | 0.210 | 0.219 | 0.219 | 0.211 |
| persistence | all | 0.713 | 0.687 | 0.709 | 0.711 |
| persistence | cropland | 0.722 | 0.674 | 0.711 | 0.715 |

## Per test year (strict, h=1), PR-AUC

| target | test_year | subset | n | clim | pers | M1: gbm | M3: gbm |
|---|---|---|---|---|---|---|---|
| occurrence | 2019 | all | 6318 | 0.302 | 0.208 | 0.250 | 0.356 |
| occurrence | 2019 | cropland | 1420 | 0.485 | 0.249 | 0.330 | 0.481 |
| occurrence | 2020 | all | 5092 | 0.429 | 0.207 | 0.222 | 0.462 |
| occurrence | 2020 | cropland | 2091 | 0.395 | 0.189 | 0.175 | 0.375 |
| occurrence | 2021 | all | 2933 | 0.408 | 0.184 | 0.257 | 0.420 |
| occurrence | 2021 | cropland | 199 | 0.230 | 0.142 | 0.119 | 0.325 |
| occurrence | 2022 | all | 2985 | 0.513 | 0.216 | 0.358 | 0.521 |
| occurrence | 2022 | cropland | 518 | 0.439 | 0.184 | 0.268 | 0.461 |
| occurrence | 2023 | all | 6024 | 0.464 | 0.204 | 0.253 | 0.485 |
| occurrence | 2023 | cropland | 1582 | 0.506 | 0.206 | 0.200 | 0.522 |
| occurrence | 2024 | all | 5193 | 0.501 | 0.190 | 0.303 | 0.515 |
| occurrence | 2024 | cropland | 2104 | 0.472 | 0.178 | 0.251 | 0.502 |
| occurrence | 2025 | all | 3287 | 0.463 | 0.144 | 0.154 | 0.484 |
| occurrence | 2025 | cropland | 213 | 0.225 | 0.112 | 0.172 | 0.204 |
| persistence | 2019 | all | 2175 | 0.853 | 0.695 | 0.729 | 0.862 |
| persistence | 2019 | cropland | 752 | 0.910 | 0.751 | 0.761 | 0.898 |
| persistence | 2020 | all | 1930 | 0.872 | 0.641 | 0.733 | 0.871 |
| persistence | 2020 | cropland | 657 | 0.865 | 0.598 | 0.687 | 0.864 |
| persistence | 2021 | all | 1124 | 0.919 | 0.728 | 0.770 | 0.927 |
| persistence | 2021 | cropland | 53 | 0.958 | 0.767 | 0.836 | 0.968 |
| persistence | 2022 | all | 1379 | 0.888 | 0.733 | 0.756 | 0.897 |
| persistence | 2022 | cropland | 190 | 0.910 | 0.616 | 0.810 | 0.927 |
| persistence | 2023 | all | 2407 | 0.846 | 0.642 | 0.659 | 0.854 |
| persistence | 2023 | cropland | 590 | 0.872 | 0.641 | 0.669 | 0.865 |
| persistence | 2024 | all | 1874 | 0.827 | 0.641 | 0.732 | 0.826 |
| persistence | 2024 | cropland | 644 | 0.848 | 0.602 | 0.724 | 0.857 |
| persistence | 2025 | all | 760 | 0.800 | 0.653 | 0.617 | 0.809 |
| persistence | 2025 | cropland | 39 | 0.745 | 0.512 | 0.648 | 0.728 |

## Operating points (strict): share of floods flagged at 5% / 10% false-alarm rate

| h | target | subset | model | recall_at_fpr05 | recall_at_fpr10 |
|---|---|---|---|---|---|
| 1 | occurrence | all | clim | 0.274 | 0.428 |
| 1 | occurrence | all | pers | 0.013 | 0.234 |
| 1 | occurrence | all | M1: gbm | 0.112 | 0.215 |
| 1 | occurrence | all | M3: gbm | 0.316 | 0.449 |
| 1 | occurrence | cropland_harvest | clim | 0.216 | 0.313 |
| 1 | occurrence | cropland_harvest | pers | 0.015 | 0.015 |
| 1 | occurrence | cropland_harvest | M1: gbm | 0.042 | 0.096 |
| 1 | occurrence | cropland_harvest | M3: gbm | 0.205 | 0.321 |
| 1 | persistence | all | clim | 0.276 | 0.466 |
| 1 | persistence | all | pers | 0.000 | 0.000 |
| 1 | persistence | all | M1: gbm | 0.119 | 0.196 |
| 1 | persistence | all | M3: gbm | 0.312 | 0.456 |
| 1 | persistence | cropland_harvest | clim | 0.338 | 0.514 |
| 1 | persistence | cropland_harvest | pers | 0.000 | 0.000 |
| 1 | persistence | cropland_harvest | M1: gbm | 0.171 | 0.275 |
| 1 | persistence | cropland_harvest | M3: gbm | 0.403 | 0.535 |
| 1 | occurrence | all | M1_physical: gbm | 0.124 | 0.231 |
| 1 | occurrence | cropland_harvest | M1_physical: gbm | 0.061 | 0.115 |
| 1 | persistence | all | M1_physical: gbm | 0.111 | 0.189 |
| 1 | persistence | cropland_harvest | M1_physical: gbm | 0.168 | 0.226 |
