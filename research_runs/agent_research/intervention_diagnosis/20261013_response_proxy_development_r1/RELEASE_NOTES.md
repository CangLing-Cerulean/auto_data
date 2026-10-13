# M1 development: completed runs, failed numerical audit preserved

27 full trainings, 27 proxy trajectories and 18 short runs completed; cumulative completed full trainings 331, with the latest 27 not accepted by the full stage audit. Original float32 identity tolerance failed and was not relaxed or rerun. Forecasts were published before the 18 full child labels.

Recorded response MAE: M1 0.407517 pp, RAW 7.845171 pp, FIXED 1.386154 pp, SHORT 1.958771 pp. These are descriptive, not fully independently accepted metrics. M1 and FIXED recommendations are identical, so the pre-registered strict recommendation increment fails regardless of labels. Official error/D gates remain null; overall advancement false.

Archive integrity only: 293,218,806 bytes, 1,573 members; SHA256 602e5519bb5b5d4f3ea8d637e836e2f68d7d02ef5eee6c73a54574f5cd0915ff. Preserves original failure log, frozen predictions, model/Adam/RNG states and derived development partitions. No base test data or credentials. No background training or automatic continuation.
