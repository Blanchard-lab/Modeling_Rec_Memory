=== Sanity Check ===
Overall label distribution:
1.0    662
0.0    651
Name: scene_familiarity, dtype: int64

Label counts by participant:
participant  scene_familiarity
1            1.0                  22
             0.0                  21
2            0.0                  24
             1.0                  19
3            0.0                  34
             1.0                  32
4            0.0                  43
             1.0                  43
5            1.0                  15
             0.0                  13
6            0.0                  33
             1.0                  32
7            0.0                  24
             1.0                  23
8            0.0                  18
             1.0                  18
9            0.0                  29
             1.0                  29
10           0.0                   7
             1.0                   7
11           0.0                   4
             1.0                   4
12           1.0                  16
             0.0                  15
13           1.0                  23
             0.0                  22
14           0.0                  45
             1.0                  44
15           1.0                  22
             0.0                  21
16           1.0                  22
             0.0                  21
17           0.0                  29
             1.0                  29
18           0.0                  33
             1.0                  33
19           0.0                  30
             1.0                  30
21           1.0                  22
             0.0                  20
22           1.0                  26
             0.0                  25
23           1.0                  45
             0.0                  37
24           1.0                  59
             0.0                  56
25           0.0                  29
             1.0                  29
26           0.0                  18
             1.0                  18
Name: scene_familiarity, dtype: int64

Participants with only one label class:
[]
=== End Sanity Check ===


=== Checking Within-Participant Label Order ===
Participant 1: first 20 labels = [0. 1. 1. 1. 1. 1. 1. 0. 1. 0. 1. 1. 1. 0. 0. 1. 0. 0. 0. 0.]
  → Are labels sorted? False
Participant 2: first 20 labels = [0. 1. 1. 1. 1. 1. 1. 0. 0. 0. 1. 1. 1. 0. 0. 1. 0. 0. 0. 0.]
  → Are labels sorted? False
Participant 3: first 20 labels = [0. 1. 1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 1. 0. 1. 1. 1. 1. 1.]
  → Are labels sorted? False
=== End Label Order Check ===

Feature Names:  ['blink_cnt', 'peak_blink_duration', 'avg_blink_duration']
Participants:  [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 21 22 23 24 25
 26]
[0. 1. 1. 1. 1. 1. 1. 0. 1. 0. 1. 1. 1. 0. 0. 1. 0. 0. 0. 0.]
✅ Leave-One-Group-Out sanity check passed — no data leakage.

=== Checking Label Distribution Per Fold ===
Fold 0 (Participant 1): Test labels = [0. 1.], Count: 43, Distribution: [21 22]
Fold 1 (Participant 2): Test labels = [0. 1.], Count: 43, Distribution: [24 19]
Fold 2 (Participant 3): Test labels = [0. 1.], Count: 66, Distribution: [34 32]
Fold 3 (Participant 4): Test labels = [0. 1.], Count: 86, Distribution: [43 43]
Fold 4 (Participant 5): Test labels = [0. 1.], Count: 28, Distribution: [13 15]
Fold 5 (Participant 6): Test labels = [0. 1.], Count: 65, Distribution: [33 32]
Fold 6 (Participant 7): Test labels = [0. 1.], Count: 47, Distribution: [24 23]
Fold 7 (Participant 8): Test labels = [0. 1.], Count: 36, Distribution: [18 18]
Fold 8 (Participant 9): Test labels = [0. 1.], Count: 58, Distribution: [29 29]
Fold 9 (Participant 10): Test labels = [0. 1.], Count: 14, Distribution: [7 7]
Fold 10 (Participant 11): Test labels = [0. 1.], Count: 8, Distribution: [4 4]
Fold 11 (Participant 12): Test labels = [0. 1.], Count: 31, Distribution: [15 16]
Fold 12 (Participant 13): Test labels = [0. 1.], Count: 45, Distribution: [22 23]
Fold 13 (Participant 14): Test labels = [0. 1.], Count: 89, Distribution: [45 44]
Fold 14 (Participant 15): Test labels = [0. 1.], Count: 43, Distribution: [21 22]
Fold 15 (Participant 16): Test labels = [0. 1.], Count: 43, Distribution: [21 22]
Fold 16 (Participant 17): Test labels = [0. 1.], Count: 58, Distribution: [29 29]
Fold 17 (Participant 18): Test labels = [0. 1.], Count: 66, Distribution: [33 33]
Fold 18 (Participant 19): Test labels = [0. 1.], Count: 60, Distribution: [30 30]
Fold 19 (Participant 21): Test labels = [0. 1.], Count: 42, Distribution: [20 22]
Fold 20 (Participant 22): Test labels = [0. 1.], Count: 51, Distribution: [25 26]
Fold 21 (Participant 23): Test labels = [0. 1.], Count: 82, Distribution: [37 45]
Fold 22 (Participant 24): Test labels = [0. 1.], Count: 115, Distribution: [56 59]
Fold 23 (Participant 25): Test labels = [0. 1.], Count: 58, Distribution: [29 29]
Fold 24 (Participant 26): Test labels = [0. 1.], Count: 36, Distribution: [18 18]
✅ All folds have both label classes!

Participant 1: labels = [0. 1. 1. 1. 1. 1. 1. 0. 1. 0. 1. 1. 1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0.
 0. 1. 1. 0. 1. 0. 1. 1. 1. 1. 0. 1. 0. 0. 1. 0. 1. 0. 0.]
Participant 2: labels = [0. 1. 1. 1. 1. 1. 1. 0. 0. 0. 1. 1. 1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0.
 0. 1. 1. 0. 1. 0. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
Participant 3: labels = [0. 1. 1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.
 0. 0. 1. 1. 0. 0. 0. 1. 1. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 1. 1.
 1. 0. 0. 1. 1. 1. 0. 1. 1. 1. 0. 0. 1. 1. 1. 0. 0. 0.]
Class weights: {0: 1.0084485407066053, 1: 0.9916918429003021}
