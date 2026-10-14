# Feature ablation: RF

Test split: 136 INCLUDE clips from recording sittings not seen in training. Values are mean ± sample standard deviation over seeds.

| Configuration | Features | Runs | Val top-1 | Test top-1 | Test top-5 | Test macro F1 |
|---|---|---|---|---|---|---|
| hands | 128 | 3 | 0.569 ± 0.026 | 0.591 ± 0.008 | 0.907 ± 0.004 | 0.552 ± 0.006 |
| hands_pose | 149 | 3 | 0.667 ± 0.026 | 0.681 ± 0.011 | 0.941 ± 0.007 | 0.630 ± 0.013 |
| hands_pose_face | 308 | 3 | 0.601 ± 0.028 | 0.593 ± 0.011 | 0.934 ± 0.007 | 0.547 ± 0.017 |

## Individual runs

| Configuration | Seed | Epochs (best) | Val top-1 | Test top-1 | Test top-5 | Test macro F1 |
|---|---|---|---|---|---|---|
| hands | 42 |  | 0.598 | 0.581 | 0.904 | 0.545 |
| hands | 43 |  | 0.559 | 0.596 | 0.912 | 0.556 |
| hands | 44 |  | 0.549 | 0.596 | 0.904 | 0.555 |
| hands_pose | 42 |  | 0.647 | 0.669 | 0.934 | 0.618 |
| hands_pose | 43 |  | 0.657 | 0.684 | 0.949 | 0.629 |
| hands_pose | 44 |  | 0.696 | 0.691 | 0.941 | 0.644 |
| hands_pose_face | 42 |  | 0.569 | 0.581 | 0.926 | 0.530 |
| hands_pose_face | 43 |  | 0.618 | 0.596 | 0.934 | 0.549 |
| hands_pose_face | 44 |  | 0.618 | 0.603 | 0.941 | 0.564 |
