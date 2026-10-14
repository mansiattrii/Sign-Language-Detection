# Feature ablation: LSTM

Test split: 136 INCLUDE clips from recording sittings not seen in training. Values are mean ± sample standard deviation over seeds.

| Configuration | Features | Runs | Val top-1 | Test top-1 | Test top-5 | Test macro F1 |
|---|---|---|---|---|---|---|
| hands | 128 | 3 | 0.817 ± 0.030 | 0.752 ± 0.021 | 0.985 ± 0.013 | 0.760 ± 0.015 |
| hands_pose | 149 | 3 | 0.863 ± 0.052 | 0.757 ± 0.060 | 0.968 ± 0.015 | 0.751 ± 0.060 |
| hands_pose_face | 308 | 3 | 0.686 ± 0.017 | 0.691 ± 0.072 | 0.922 ± 0.008 | 0.687 ± 0.093 |

## Individual runs

| Configuration | Seed | Epochs (best) | Val top-1 | Test top-1 | Test top-5 | Test macro F1 |
|---|---|---|---|---|---|---|
| hands | 42 | 165 (115) | 0.784 | 0.765 | 0.971 | 0.765 |
| hands | 43 | 120 (70) | 0.824 | 0.728 | 0.993 | 0.743 |
| hands | 44 | 134 (84) | 0.843 | 0.765 | 0.993 | 0.771 |
| hands_pose | 42 | 107 (57) | 0.824 | 0.706 | 0.956 | 0.703 |
| hands_pose | 43 | 164 (114) | 0.922 | 0.824 | 0.963 | 0.819 |
| hands_pose | 44 | 133 (83) | 0.843 | 0.743 | 0.985 | 0.732 |
| hands_pose_face | 42 | 175 (125) | 0.696 | 0.772 | 0.926 | 0.786 |
| hands_pose_face | 43 | 76 (26) | 0.667 | 0.632 | 0.926 | 0.601 |
| hands_pose_face | 44 | 154 (104) | 0.696 | 0.669 | 0.912 | 0.675 |
