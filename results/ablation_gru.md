# Feature ablation: GRU

Test split: 136 INCLUDE clips from recording sittings not seen in training. Values are mean ± sample standard deviation over seeds.

| Configuration | Features | Runs | Val top-1 | Test top-1 | Test top-5 | Test macro F1 |
|---|---|---|---|---|---|---|
| hands | 128 | 3 | 0.791 ± 0.091 | 0.694 ± 0.011 | 0.949 ± 0.013 | 0.697 ± 0.017 |
| hands_pose | 149 | 3 | 0.840 ± 0.023 | 0.679 ± 0.030 | 0.968 ± 0.011 | 0.677 ± 0.048 |
| hands_pose_face | 308 | 3 | 0.673 ± 0.025 | 0.676 ± 0.000 | 0.931 ± 0.011 | 0.674 ± 0.028 |

## Individual runs

| Configuration | Seed | Epochs (best) | Val top-1 | Test top-1 | Test top-5 | Test macro F1 |
|---|---|---|---|---|---|---|
| hands | 42 | 151 (101) | 0.765 | 0.706 | 0.941 | 0.717 |
| hands | 43 | 218 (168) | 0.892 | 0.691 | 0.941 | 0.687 |
| hands | 44 | 108 (58) | 0.716 | 0.684 | 0.963 | 0.689 |
| hands_pose | 42 | 170 (120) | 0.853 | 0.684 | 0.971 | 0.694 |
| hands_pose | 43 | 119 (69) | 0.853 | 0.647 | 0.956 | 0.623 |
| hands_pose | 44 | 142 (92) | 0.814 | 0.706 | 0.978 | 0.714 |
| hands_pose_face | 42 | 128 (78) | 0.647 | 0.676 | 0.919 | 0.642 |
| hands_pose_face | 43 | 151 (101) | 0.696 | 0.676 | 0.941 | 0.695 |
| hands_pose_face | 44 | 144 (94) | 0.676 | 0.676 | 0.934 | 0.685 |
