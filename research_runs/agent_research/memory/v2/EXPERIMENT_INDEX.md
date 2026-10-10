# 历史实验索引

134次历史训练（131主预测、3辅助）及10条条件性摘要。按ID排列，不按效果推荐方向。

| 实验 | 配方/模型 | 评价范围 | NMSE | NMAE | 原结果SHA256 |
|---|---|---|---:|---:|---|
| E000_baseline_s2026 | baseline/original_DLinear | development_validation | 0.2859968726100614 | 0.1992694365875478 | 62561b127df567af2ea56f8217c75384b5cf84e779a496a233d248439cfde482 |
| E001_last_center_s2026 | last_center/original_DLinear | development_validation | 0.30441882617924976 | 0.1815340482898374 | 8b82e210db8c3bcb54a6be593b2a2575926677a3e5f6b4e9447ef4bb9e93b6b2 |
| E002_mean_center_s2026 | mean_center/original_DLinear | development_validation | 0.30519630604198766 | 0.18642378460849565 | 17b914e0b0d4b17b5337b8cb56486516c7445b63f432b0defa106af774d126e2 |
| E003_selective_center_s2026 | selective_center/original_DLinear | development_validation | 0.28370452890649894 | 0.19516159217652293 | 9bba599fd060b7a69cca124d469138f12272f53e395617ae217d2fd8a37a43f2 |
| E004_baseline_s2027 | baseline/original_DLinear | development_validation | 0.2896458414567915 | 0.19517469445877747 | 333ec67420714d67a68cf12adbbd70cec1b2e83779c0935904b419d8fcd62a6c |
| E005_selective_center_s2027 | selective_center/original_DLinear | development_validation | 0.287753899079629 | 0.19155373617534582 | 44b015f5dd3742798638f3ebad4127295553acaac315dc0f0f5a3cb40ba12ce9 |
| E006_baseline_s2028 | baseline/original_DLinear | development_validation | 0.28837501139300636 | 0.1956313281921909 | 1529048a83b148ff4237cc67716e1f5d8ed3bf068eaa261b2ecdc22ed5cc98f5 |
| E007_selective_center_s2028 | selective_center/original_DLinear | development_validation | 0.28696629469521767 | 0.1922662354838746 | 59ef80f06a132fcfdb709ac2ce9f0bc1493174a3ea92f826b24be766e89bd711 |
| S000_uniform_s2026 | uniform/original_DLinear | development_validation | 0.2880167633442773 | 0.19500733246096835 | f2a42f47cdc622eeaea15a756ed6743c60e5e5fcc6b016b0fee0fbc88d8588a5 |
| S001_drop_quiet_s2026 | drop_quiet/original_DLinear | development_validation | 0.28782575427271634 | 0.19064831433697974 | 935419305c49a547705881add81aa2d174b68600d18b0c76b064099944ec8aed |
| S002_downweight_quiet_s2026 | downweight_quiet/original_DLinear | development_validation | 0.28940913618765574 | 0.19122074207413028 | 80d2aa7a0fea40251fd954d52ddb6450421c12b269a1e0e4b9ab0c3f66546563 |
| S003_random_drop_s2026 | random_drop/original_DLinear | development_validation | 0.28744768764631157 | 0.19583653504402232 | b9e1d688a092b94cd01687c8469c2000f61e45914e5a1d9228feca16f6b7dd64 |
| S004_oversample_high_s2026 | oversample_high/original_DLinear | development_validation | 0.2910870865062728 | 0.19378296366893286 | 551a6ad0122614a0020f95d2a3573af3268d19556f3afc89e18a6bd66d4217e7 |
| S005_uniform_s2027 | uniform/original_DLinear | development_validation | 0.28843843433606425 | 0.1980166072475031 | 9e995ed5326dadd6b8e4f9f1689b487a6a449e7ba876b4e91989c33c39b7e6c7 |
| S006_drop_quiet_s2027 | drop_quiet/original_DLinear | development_validation | 0.2885317223519248 | 0.19353847604056806 | b8b6c4eb7d3fa27ad461f39e9f39e2c92839a0ce9db362b51654bb2ebce599bf |
| S007_downweight_quiet_s2027 | downweight_quiet/original_DLinear | development_validation | 0.28937694480161325 | 0.194848602290598 | e33db2ab425a67ed590b9375244d8eaea8b25bb38c8243e9a1064afde0f7c8e8 |
| S008_random_drop_s2027 | random_drop/original_DLinear | development_validation | 0.29076207640001633 | 0.1952918497264439 | f11a3c950ba90170c52c12c4abb39427fe5364f9cb17a59a1c95921dbe730ae0 |
| S009_oversample_high_s2027 | oversample_high/original_DLinear | development_validation | 0.29075422112449606 | 0.1949896512276228 | 26be556b96fbe715a4b2fcd5c491e74ace8412af757f31f0bdf4cf4a03b1cceb |
| S010_uniform_s2028 | uniform/original_DLinear | development_validation | 0.2868006574591469 | 0.19582507235031132 | 48f4f039a08565bb1ad70307aa6bcfd90e8573600844847b3925a0b2503ac74e |
| S011_drop_quiet_s2028 | drop_quiet/original_DLinear | development_validation | 0.28905670108347564 | 0.1901179607424941 | ca75f3c9c89a65ee4713a3db64019e521d00a2e2d9b564aaef260b3fb19307e1 |
| S012_downweight_quiet_s2028 | downweight_quiet/original_DLinear | development_validation | 0.2881711517186182 | 0.19203207316487106 | beada6166008590b3ac34a094c205b32e9a744c04ece776ccb4af27d3875d9c0 |
| S013_random_drop_s2028 | random_drop/original_DLinear | development_validation | 0.28807662421836855 | 0.19343828663011894 | 4e29351f747d62053908044a9d25378b7c436fbc20136550bd8165fd22f98704 |
| S014_oversample_high_s2028 | oversample_high/original_DLinear | development_validation | 0.29127067731943485 | 0.19065171257346417 | 14e00ffb27d8c40171befbf557be8fc9f9897660a07d58563db0c92f5da1e1df |
| S015_month_matched_drop_s2026 | month_matched_drop/original_DLinear | development_validation | 0.28878003427207244 | 0.19583832323779293 | 5aeee4bb3cc43b1afec795c5de1202b4c2f93d484ce29621dcab7b439308c371 |
| S016_month_matched_drop_s2027 | month_matched_drop/original_DLinear | development_validation | 0.28864722883226657 | 0.19925442784455932 | 45f117723dccc6de2a27483b22fc9241575c7306fa6c117262a07ff137177693 |
| S017_month_matched_drop_s2028 | month_matched_drop/original_DLinear | development_validation | 0.2910432001859124 | 0.19609494653783183 | 8af85e55ca0337507948fa522b0a39785ae2984255838578f408a6cbbaea05a6 |
| S018_protect_lps_matched_s2026 | protect_lps_matched/original_DLinear | development_validation | 0.2891224747871388 | 0.194852403188638 | 1b048dadf8d2735a61116c03881e64cc71a3835482fc001ef9b1fb47c23a1121 |
| S019_placebo_matched_swap_s2026 | placebo_matched_swap/original_DLinear | development_validation | 0.2874200876744733 | 0.19215868824537818 | b461c2cb22e664fa9073f6ac240eaedeaf1bf097619200f59629c119208f35a7 |
| S020_protect_lps_matched_s2027 | protect_lps_matched/original_DLinear | development_validation | 0.2884422119884007 | 0.19431120521307735 | e5a58106e3986281d56d8b9444e949cad6654599eb25794fb0bfb7818cebd071 |
| S021_placebo_matched_swap_s2027 | placebo_matched_swap/original_DLinear | development_validation | 0.28810820757528793 | 0.1940791851766596 | f75cfcda77af4330b58774b7fa84053118cf9851fc0b0dcf1c91fdd752240e98 |
| S022_protect_lps_matched_s2028 | protect_lps_matched/original_DLinear | development_validation | 0.2883326899342955 | 0.19421290695523707 | 5ff2e4e308a01496b580b4007834765918a4fb152fd74741021e572109a239bd |
| S023_placebo_matched_swap_s2028 | placebo_matched_swap/original_DLinear | development_validation | 0.2898343327469551 | 0.19525022991517135 | 86641bf1fa8439aefe7f3f91a61e1e1506125a2c010e136b8a7f00b3f7835e18 |
| T000_prefix60_full | T000_prefix60_full/original_DLinear | train_internal_forward | 0.4477611067420934 | 0.21474054497371273 | 2fe1654a750b055e190d39ee1a5f559374dff4719f8948cf6345c3880bf10147 |
| T001_prefix60_drop0 | T001_prefix60_drop0/original_DLinear | train_internal_forward | 0.44305459023029636 | 0.21506010875388556 | 244a3a33e4356148ecc8e945ce9b6cb3a3fee2ed34c02b50353ef4cc2b0dc313 |
| T002_prefix60_drop1 | T002_prefix60_drop1/original_DLinear | train_internal_forward | 0.4434554681249404 | 0.2135003501156006 | 1426796a4e976a8fd1bc89ba1ddc229dfda3967a8cd53c04548ceea0c335d6ce |
| T003_prefix60_drop2 | T003_prefix60_drop2/original_DLinear | train_internal_forward | 0.44647448357090974 | 0.21593409458661514 | 00870401c57ac129a9b8b4790d17234aea7d7c03b6a3ac5c15acc8601cb45f31 |
| T004_prefix60_drop3 | T004_prefix60_drop3/original_DLinear | train_internal_forward | 0.5221368287186365 | 0.22230058711123993 | 3afac28478b94e163b80ff0848b1ccf3317662371acee4185ca493bb834c07a0 |
| T005_prefix80_full | T005_prefix80_full/original_DLinear | train_internal_forward | 0.5300340324423901 | 0.2542354034596576 | 6d44c16cbc860634f00c0dcce24aa0f301aadd13edf2f8b436b3d92c0e5ad104 |
| T006_prefix80_drop0 | T006_prefix80_drop0/original_DLinear | train_internal_forward | 0.5253654952766813 | 0.2507126917194915 | 0e352a77d54a58e6c97ef4da66e0dc00f7d278cac0638d69cd09393e8654d860 |
| S024_future_conflict_downweight_s2026 | future_conflict_downweight/original_DLinear | development_validation | 0.2992177902266913 | 0.18367938293224573 | 6555ef3a4a9aa079688b2f91e5695255cdcfe3d7b5ab19b5f8e9700a66ffb0fb |
| S025_future_conflict_shuffled_s2026 | future_conflict_shuffled/original_DLinear | development_validation | 0.2888611463184397 | 0.19572374543798024 | 3f3b9e9be254de32fff0af07e7ab867dcb43a432690ad174ad57c9a7807ae209 |
| T007_prefix60_disjoint | T007_prefix60_disjoint/original_DLinear | train_internal_forward | 0.46789805665462453 | 0.2198121582478655 | 2017b822b333d595e95f6b5aff8ac2e0803c82288cc1c26920f3db9534a53118 |
| T008_prefix60_matched_random | T008_prefix60_matched_random/original_DLinear | train_internal_forward | 0.4701455319275894 | 0.22302704263195725 | d62335e2b8da13fc1c80157ed27aaf8ad374467ed4463a77865785741ec7ad90 |
| I000_adaptive_is_s2026 | adaptive_is/original_DLinear | development_validation | 0.28616945674996563 | 0.19420091041550458 | 675323372a02a8afcfee96331568c9112ff8341b2f48f1331befb4dbf92e163e |
| T009_prefix60_state_balance | T009_prefix60_state_balance/original_DLinear | train_internal_forward | 0.44680528816337445 | 0.21203294374458376 | 5685fbd89df4137ae6ac8a2bc6548550641ad8cd2f37a2c4591bd94c2c2e232d |
| R000_pca_rotation_s2026 | pca_rotation/original_DLinear | development_validation | 0.28990032687171896 | 0.2273284960613014 | b5912bdb4bbf8b7525c31f2ddd38079bd5598471fa501bfedae62e58d4e892cb |
| I001_adaptive_is_numpy_cdf_s2026 | adaptive_is/original_DLinear | development_validation | 0.28662138100312534 | 0.1953365172722955 | 0e19b2e1a70a21b72d773a77a096c76962b486cb45756876630bbc23d734d0c1 |
| T010_prefix60_sources | T010_prefix60_sources/original_DLinear | train_internal_forward | 0.43296964251281966 | 0.20957066547403777 | d5bc90930648f11ee50eb30ef03d905ac4cb21bb909960bf2cd6a2d844d8f1bb |
| T011_prefix80_sources | T011_prefix80_sources/original_DLinear | train_internal_forward | 0.51711211026981 | 0.25250101829349164 | eec24ff0f2a113b9e2d64a7895b72d7851d220bcc99f2ad5b50c97d43c8e94d1 |
| R001_sources_s2026 | leading_sources/original_DLinear | development_validation | 0.2736735069613985 | 0.19863202164921018 | 268faa528fedb7327876c5b0547fe5254c5dd7795485e70f1287a18314bff220 |
| T012_prefix60_discordance | T012_prefix60_discordance/original_DLinear | train_internal_forward | 0.4330261551169938 | 0.20897782783050717 | bf99b383213bcd0d40aa5674f52a2d37d021edd113f21a8f0a63b7140efb654e |
| T013_prefix60_discordance_shuffle | T013_prefix60_discordance_shuffle/original_DLinear | train_internal_forward | 0.43299827316366346 | 0.20937990313472354 | 4ecf4713fb89a78f742a9a90ce64f67900df9f5dbe22d920fe9247951f1c8954 |
| T017_prefix60_cluster1 | T017_prefix60_cluster1/original_DLinear | train_internal_forward | 0.5539915285515151 | 0.2605168624223245 | 98fd6db5934d2af764881ded90779d2840128e6e882cf59024300aa1bae2a1e6 |
| T018_prefix60_cluster2 | T018_prefix60_cluster2/original_DLinear | train_internal_forward | 0.5996915238388406 | 0.2697651849132395 | 7d693b772cd3316bf5f9dce24ef3fd8c279f19b3763b5ceb5f61e815c07971f9 |
| T019_prefix60_cluster3 | T019_prefix60_cluster3/original_DLinear | train_internal_forward | 0.522990282695478 | 0.2228847484635604 | ad0b3f94f3bb2ff1399a4dfca87181d281729f1be4c1d392eaa5b844a738dd1f |
| T023_prefix60_balanced | T023_prefix60_balanced/original_DLinear | train_internal_forward | 0.44540008262473674 | 0.20922832369913028 | 20aa5e569bf38cacd78e44537e1475699642d98b2f5c45be5e3091d0627fe83f |
| T024_prefix60_recent | T024_prefix60_recent/original_DLinear | train_internal_forward | 0.445087268272442 | 0.21763150111029336 | cb4aa3a0c7e6dc7f2c5ecb04208100617a3240248545f6ee712d3fb0d65b1b55 |
| T025_prefix60_balanced_recent | T025_prefix60_balanced_recent/original_DLinear | train_internal_forward | 0.4442564755823463 | 0.20902297121631827 | 7e1528647f7805c203ad26f2192ac8174ac0406ad7a148aebbff213cca2a2d2e |
| T026_prefix60_recent_shuffle | T026_prefix60_recent_shuffle/original_DLinear | train_internal_forward | 0.44453175930039174 | 0.2194491908360776 | 0f6af282682575e215a19120371ecfb89bf534d165b30971a6c07763e8ad436a |
| T027_prefix80_balanced_recent | T027_prefix80_balanced_recent/original_DLinear | train_internal_forward | 0.533201283712393 | 0.24448878588379577 | 757e7c6d6a09cb8f791ab2e367bb032d5dcf74646e19e5d2d2bdaa0698968765 |
| T028_prefix80_recent | T028_prefix80_recent/original_DLinear | train_internal_forward | 0.5360383491434576 | 0.2520850540175993 | e542d2739c8178e89e5aebbaea694b55f3ef7660bcce43c4f5962116ce6fe977 |
| T029_prefix80_recent_shuffle | T029_prefix80_recent_shuffle/original_DLinear | train_internal_forward | 0.5283682114655508 | 0.2507089834783883 | f5115a71a5adbb7be240709017326ac78459c9a1b06bb79ad89c5b03964d4977 |
| T030_prefix60_learning_progress | progress/original_DLinear | train_internal_forward | 0.8007784355265272 | 0.5053725454680031 | afcb358877a7a1ff9e3cd13fae986570ab5338de45aeec2a6adafd85cf20ad73 |
| T031_prefix60_learning_hard | hard/original_DLinear | train_internal_forward | 0.4613730549402339 | 0.27137105307427006 | 83a438685c354e79df9733713a55cf0e0c36924160872406abb9138e741a8109 |
| T032_prefix60_learning_shuffle | shuffle/original_DLinear | train_internal_forward | 0.4436048973712523 | 0.21657978733628339 | de6cefd2b6005f0dfc0dedef36b2df703c45143d5aab5549348e527e2312c47c |
| P000_reference_prefix60 | reference/auxiliary | training_auxiliary_no_forecast_evaluation | 未提供 | 未提供 | ed85826d689e6f8de72103dcd7ff585882d9bc6d426c3d8ed583680a00ec0397 |
| T035_prefix60_rho | rho/original_DLinear | train_internal_forward | 0.4703107694934512 | 0.22346295003966257 | 73d8eab643a55d5c12685ea4546bedb5605a2a9ee3dcdb1f09afd8a1345dcee6 |
| T036_prefix60_adarho | adarho/original_DLinear | train_internal_forward | 0.4815199667684456 | 0.1800732917658601 | 9549177dde6f290c5054e86285e10dfcc8622e798d9f561ebdccab3e78061ce8 |
| T037_prefix60_hard_rank | hard_rank/original_DLinear | train_internal_forward | 0.45347623940175275 | 0.26146947216139405 | 87d95ec5235c603e3ae2f0280052c18712ced872b9bb618e798f54e676d6609a |
| T038_prefix60_shuffle | shuffle/original_DLinear | train_internal_forward | 0.4484471605967026 | 0.20961662477767087 | cdd77d3b3507e3b9877830eb30bea0cf5c45bc493f3d6d0ff8e43eec3d39c4cf |
| T042_prefix60_reshuffle | reshuffle/original_DLinear | train_internal_forward | 0.4458772492810256 | 0.22098981315954402 | 4fa4e77e5657ee90e267a1fe37cfa7ec4cfecb2b072b09a8cec80ec8a52e8b1d |
| T043_prefix60_time_stratified | time_stratified/original_DLinear | train_internal_forward | 0.44477964481036014 | 0.22000705069465443 | 43241a23c3ce347ccce74c83a754c565dfe3bf7c4afce1c6de713141179799dc |
| T044_prefix80_time_stratified | time_stratified/original_DLinear | train_internal_forward | 0.5343100911180593 | 0.25902814139609087 | 1e4fe50689eb539f5fe102c792ea4c1cdd6b4989200a1e5f66a6980d385e1b19 |
| P003_covariate_prefix60 | P003_covariate_prefix60/auxiliary | training_auxiliary_no_forecast_evaluation | 未提供 | 未提供 | 5d140b38428d45bbe431b6712bb37dab12116fafd09a140f35bd36f084daf312 |
| T045_prefix60_covariate_global | global/original_DLinear | train_internal_forward | 0.44504730282227367 | 0.2116944067571408 | 1caeeb60aacd7a963f455524e3a2e7b550868ae5aefa8b0ef140990dac4e9255 |
| T046_prefix60_covariate_state_matched | state_matched/original_DLinear | train_internal_forward | 0.4511091236948079 | 0.21033449870323345 | f5531bd26bb3b2d0b5735e49608fedd917058f90d47f9a8c08af0de84505814f |
| T047_prefix60_covariate_shuffle | shuffle/original_DLinear | train_internal_forward | 0.4458887133717834 | 0.21054583141605415 | fb6a04fd2be0f323dd8f2c8eb10268bb29a84cb7da7d3d5ca6e8379567f161ba |
| P004_covariate_prefix80 | P004_covariate_prefix80/auxiliary | training_auxiliary_no_forecast_evaluation | 未提供 | 未提供 | ca62f8f6d419bc4bef50ea6fb8e4adcc91091dff226fd9a36ffb298976b42d9c |
| T048_prefix80_covariate_global | global/original_DLinear | train_internal_forward | 0.5414443128763474 | 0.2583218357119067 | 88d19da2087993aa5f35cabd90a39ce1cab509cdbc7d769fb8d91db4ebb51b84 |
| T049_prefix80_covariate_shuffle | shuffle/original_DLinear | train_internal_forward | 0.5393574474508311 | 0.2537324163008065 | 58c4bfcd715ba42d3a2b870f0e2a481ca4911f58e857edd134baf670415312a9 |
| T050_prefix80_balanced | balanced/original_DLinear | train_internal_forward | 0.5274009644015832 | 0.24285059667988307 | 5311d4a2bb9a067fcea0174c64826a4739038b939279430b3a9efaeafb7efde1 |
| T051_prefix80_reshuffle | reshuffle/original_DLinear | train_internal_forward | 0.5297077037658069 | 0.2558697757227722 | ec1d16b935d57b2a175ec95e1c05fd5522327dfe0ce9cab7d00ab171636415f4 |
| S041_support_balanced_s2026 | balanced/original_DLinear | development_validation | 0.3038753346880713 | 0.19799573221888572 | 2de3b2119261bd1e78fdb03f6bda8d3e82b0b6f1fb144df93c59952a8930fbe5 |
| M000_cross_linear_s2026 | M000_cross_linear_s2026/cross_linear | development_validation | 0.2248700266876872 | 0.20682536877360247 | 7d619c813d0cc318c2b3cca09ee76f616bfd9d44022e3d955e77870949c076ae |
| M001_residual_mlp_s2026 | M001_residual_mlp_s2026/residual_mlp | development_validation | 0.17797661580592528 | 0.16321519701676845 | bf44b0d1d50dc911d51f8e267809ea771d7647628aae392ff1cb2d37d1ecb3af |
| M002_residual_mlp_s2027 | M002_residual_mlp_s2027/residual_mlp | development_validation | 0.17644374583163536 | 0.15755413901979978 | 894204590bbc39c3031a6e33c373b3f2d3e199e1afe1d15314f24598416e8b87 |
| M003_residual_mlp_s2028 | M003_residual_mlp_s2028/residual_mlp | development_validation | 0.16813769082309105 | 0.15480948114956528 | a514aeb822bfc394a03123b5820c58864b03106a3bd31f9988ecd023712f5ff9 |
| T052_prefix60_cadence_drop | drop/original_DLinear | train_internal_forward | 0.5223128764524535 | 0.21766118122301956 | 7658f7a7d0edc360df04ce448161145dfa40e6a175dbd8c5904fd6b63f5df9bc |
| T053_prefix60_cadence_soft | soft/original_DLinear | train_internal_forward | 0.45821619907752525 | 0.21377200547527217 | 0237fa5a7dd011009bc56ab613d4151daefabc4158b7b83902f0eb8d6a52f636 |
| T054_prefix60_cadence_shuffle | shuffle/original_DLinear | train_internal_forward | 0.4578032785746888 | 0.21372882541248647 | 224e3572ea7dc093be3d5fdbc4eb5252da3ca2ef5b99588faf21d3591c3173b0 |
| T057_prefix60_leverage_uncorrected | uncorrected/original_DLinear | train_internal_forward | 0.4488160267370609 | 0.21802636646141701 | 41116c1c02d3b2f69898e978e92d06bf07150e79159b7754f34667132d17aaea |
| T058_prefix60_leverage_corrected | corrected/original_DLinear | train_internal_forward | 0.44387807912058863 | 0.21744657916980228 | 775a7e37d165f8696d3724af9221f6777d936ed44f39c1a5320ae22551de3402 |
| T059_prefix60_leverage_shuffle | shuffle/original_DLinear | train_internal_forward | 0.44547244496727506 | 0.2181699689289591 | d2aeb5e106ef2a96a830e7f215debed2299c1bed24238815bf912bc75d0f7a0f |
| T060_prefix80_leverage_corrected | corrected/original_DLinear | train_internal_forward | 0.533764227644945 | 0.25627527479314927 | beecefca38ba97611e555154e216faf053cc01a1554d51d705679b0a231f411e |
| T061_prefix80_leverage_shuffle | shuffle/original_DLinear | train_internal_forward | 0.5328287659568345 | 0.2548052018247889 | ad720ac6376cf35d0dd69acfc4de80be544f57a85c48490a679dc0d35fa5bf92 |
| C000_executor_uniform_s2026 | uniform/original_DLinear | development_validation | 0.2880167633442773 | 0.19500733246096835 | 023a897d76a29d1c00aeb0ff28cd5c33f22488260c8f288e6cfeadfcef670b1c |
| C001_drop_earliest_s2026 | drop_earliest/original_DLinear | development_validation | 0.29065769036505296 | 0.19257102188332947 | 459d908da1d09b1f61d0c63320727b3753cd2d9df4e657bd6eeddcc3cb311bb4 |
| C002_drop_earliest_s2027 | drop_earliest/original_DLinear | development_validation | 0.29052028307535815 | 0.1973240465971878 | 6f294d195bbcd994a1db7d1838e4389bb9a1338bd548fd186a8d97d3bb72b14c |
| C003_drop_earliest_s2028 | drop_earliest/original_DLinear | development_validation | 0.2911450104987988 | 0.1946968956612108 | 120d4daff089887592e8162abc117b1cefd4c417c371eb474b2c4bbcc9e7cd33 |
| C004_adaptive_is_s2027 | adaptive_is/original_DLinear | development_validation | 0.2868514671767934 | 0.19697320463459927 | b1e38707a28c178e16c2b770b9eab5d601d66cdb60aead91cdf404561aea85f0 |
| C005_adaptive_is_s2028 | adaptive_is/original_DLinear | development_validation | 0.2867542967073257 | 0.1916023913849311 | 74534e68968e4dbbf6dc65b43f8b83ced0b4cb29ac645e51bfd24c33ef074723 |
| C006_reshuffle_s2026 | reshuffle/original_DLinear | development_validation | 0.286988414966223 | 0.19624626451984273 | 4a419c2f1ba177383208666372e34e28f305167e7d520217be00cfc65aeaf49a |
| C007_reshuffle_s2027 | reshuffle/original_DLinear | development_validation | 0.2883702175097313 | 0.19129934035516682 | 46093db9d8f235906c54457457e318fca2fffeca3fba0c5a0ea6dfb3c915fca2 |
| C008_reshuffle_s2028 | reshuffle/original_DLinear | development_validation | 0.28767900151963594 | 0.1960490720780535 | 488a99b80ea62849e3de2b64d3b8792df234a4e7282cee1fe8af65ac686acd1e |
| C100_prefix60_oof_window | oof_window/dlinear | train_internal_forward | 0.4527851408730493 | 0.1992904117923422 | dcd05ce05068775a8c343c31041b616b9d23e367f5166d02ab762d98437e8f94 |
| C101_prefix60_oof_window_shuffle | oof_window_shuffle/dlinear | train_internal_forward | 0.4488253204138063 | 0.2149542944664894 | ac1aaeb8c7dc675b02c99a87bee376c9e0c577963e8b79efc5197e678e9d6005 |
| C102_prefix60_oof_target | oof_target/dlinear | train_internal_forward | 0.45823802222964555 | 0.1984116606180473 | 1ebb8850476e655ff2ad96affe306b375d5663c7abe3cb0f98c92c92f7786180 |
| C103_prefix60_oof_target_shuffle | oof_target_shuffle/dlinear | train_internal_forward | 0.44779799221906363 | 0.214106285775709 | 33c1c6c67d579a1dcbd3f5c1b9de906aae3c3752997a01dbee1c9120dca01cc3 |
| C104_prefix60_bilevel | bilevel/dlinear | train_internal_forward | 0.4453954034093931 | 0.21620077861684425 | a4251de2286676c20c8537f13e64a2516f670ee203e23388e9786de370ca97af |
| C105_prefix60_bilevel_shuffle | bilevel_shuffle/dlinear | train_internal_forward | 0.44588702011554143 | 0.21729279769525725 | 22c38cb7ef97c235c51913d9003e8ce7baf38c2762d4f6f512913e85d0132bf0 |
| C106_prefix60_coverage_core_20261021 | coverage_core_20261021/dlinear | train_internal_forward | 0.4480646918849503 | 0.21805599663160072 | af4f2be8428b92770de797efb37cd85494e63cd61a663fccb6e1c93b70ba48a4 |
| C107_prefix60_supervised_core_20261021 | supervised_core_20261021/dlinear | train_internal_forward | 0.4473091997333166 | 0.21537070508017211 | 1f1e18f264849206db0670e97534678c2de8f9fe44630417f571c3fb9de24b77 |
| C108_prefix60_coreset_random_20261021 | coreset_random_20261021/dlinear | train_internal_forward | 0.44754979867734707 | 0.21465818829266492 | 3af6ed636a2211bafa03b4038c611430b41b9fb92d990b97a1d86af540ce9a5c |
| C109_prefix60_coverage_core_20261022 | coverage_core_20261022/dlinear | train_internal_forward | 0.4475463317631134 | 0.21760654650955677 | d6ea3f6d4f99127f67c695dfe18b3398ca0f5b99a47b0d665f84808f683f256c |
| C110_prefix60_supervised_core_20261022 | supervised_core_20261022/dlinear | train_internal_forward | 0.44778698329612826 | 0.21692567320097791 | 817779b863fb1c48da099840eae4fd7c67816f209ea75d4d370089d3a4d0d2e8 |
| C111_prefix60_coreset_random_20261022 | coreset_random_20261022/dlinear | train_internal_forward | 0.44758496861071045 | 0.218074941492095 | 0487f2017a7057760e87efccb6c09855a69c7f01c2a788e8f2ad1bcc37af38a5 |
| C112_prefix60_coverage_core_20261023 | coverage_core_20261023/dlinear | train_internal_forward | 0.44714851256905697 | 0.21720688538628516 | f555dfe2bc417fe7835b59a7db72ddcecc43b0c19c0f9e48acf62b1ddbc27bd0 |
| C113_prefix60_supervised_core_20261023 | supervised_core_20261023/dlinear | train_internal_forward | 0.44765808648886796 | 0.21552849828772527 | 996bd6660613bd176165835eaf48012028503fe5acfa66fa9deb1c250e9037b4 |
| C114_prefix60_coreset_random_20261023 | coreset_random_20261023/dlinear | train_internal_forward | 0.4474909474322882 | 0.21556269965699124 | 1418903c2c4a163aa6cee6fdb8e5b7770d361e6d5def18414622f96bf1694557 |
| C120_prefix80_oof_window | oof_window/dlinear | train_internal_forward | 0.5306566566169999 | 0.23889786349844855 | 271fed622cdeba0fddfe5eedad40b5af3799d0c10b10d39b439d5ad8fd24e3f6 |
| C121_prefix80_oof_window_shuffle | oof_window_shuffle/dlinear | train_internal_forward | 0.5305299670921382 | 0.2545684422195377 | 1485d9c9430fbe78a453547d72d7c0eed8d67c27f62d75c695cd962aaef575d9 |
| C122_prefix80_oof_target | oof_target/dlinear | train_internal_forward | 0.5326069054266314 | 0.23568501926789281 | 5de5e7dff5122fbbf090298c8d881e142b75d0c30bd896345414b5b7293d6035 |
| C123_prefix80_oof_target_shuffle | oof_target_shuffle/dlinear | train_internal_forward | 0.5299233221054783 | 0.25432522376557926 | ac4c365612e2bc5c58ab3e7a45c9cd8f5847587b04dee257dbc33a3134736d5b |
| C124_prefix80_bilevel | bilevel/dlinear | train_internal_forward | 0.5269859224234841 | 0.2505130250589164 | a89726420758e33c11d9ba6857f727c5b490c71a6eb4d00a054be0fee8359340 |
| C125_prefix80_bilevel_shuffle | bilevel_shuffle/dlinear | train_internal_forward | 0.527309819981656 | 0.25050468566173434 | a77330a4638dc27eeb2bb79c925d47932a6f2ad83eb7fffc50519f3696e5f7a8 |
| C126_prefix80_coverage_core_20261021 | coverage_core_20261021/dlinear | train_internal_forward | 0.5297959257488553 | 0.25499123040440486 | 91397f82dd708e839dfe88df1ad6822864be68e8cfc4c34a9a681b68220a3d7b |
| C127_prefix80_supervised_core_20261021 | supervised_core_20261021/dlinear | train_internal_forward | 0.529275113658421 | 0.2538672999542761 | c57abb82544f85290e7c7724c17bc777a2df004ad3efe1db2a72db162558bc60 |
| C128_prefix80_coreset_random_20261021 | coreset_random_20261021/dlinear | train_internal_forward | 0.5296666880104488 | 0.25212151716651365 | 95aca189ce7df09fcaffbf5f3d72e1b0e5637bcc1fd157bda1d778bc1b898d07 |
| C129_prefix80_coverage_core_20261022 | coverage_core_20261022/dlinear | train_internal_forward | 0.5306759586007704 | 0.25354442949628886 | 15d00056290bddcca5cf77d0b1dea4bb751779680acb06598414392afd3acd70 |
| C130_prefix80_supervised_core_20261022 | supervised_core_20261022/dlinear | train_internal_forward | 0.5304365120006219 | 0.2521287172256299 | 0310fa3348723683ec140ccf9d4b002d795758e7f8b1a6e7a1aa76f480280d21 |
| C131_prefix80_coreset_random_20261022 | coreset_random_20261022/dlinear | train_internal_forward | 0.5305445308776238 | 0.2538654522290205 | 0cafe8474f08ec85587bccd2147d4d30d57a5774d76640b3f5d4ea721447ece5 |
| C132_prefix80_coverage_core_20261023 | coverage_core_20261023/dlinear | train_internal_forward | 0.5308218894781562 | 0.25543252315253956 | 74800ba8d665392ca0ff8247169391368d6c3c42061cff6ce488293b02888849 |
| C133_prefix80_supervised_core_20261023 | supervised_core_20261023/dlinear | train_internal_forward | 0.5298549853683634 | 0.2533383783684881 | 65b02e20a946d96e6f65fb56f75b3fa23ecf71248324c2664960abd26d27dc25 |
| C134_prefix80_coreset_random_20261023 | coreset_random_20261023/dlinear | train_internal_forward | 0.5310115226422291 | 0.2554199639657629 | 650abbbb4555b14e9bbeba6057a8a6778a64f70d57cdef560edc2a6b97ed1ed9 |

## 条件性摘要

- experience:centering_feedback：全通道中心化失败后，通道条件方案有小幅开发收益；反馈可否定一般性假设，仍不能证明物理机制。 范围限制：通道由开发反馈选择；三个种子不能代替独立时间评价。
- experience:quiet_selection：低变化删除和降权未建立稳健优势；随机控制与局部组误差能够反驳简单的数据价值解释。 范围限制：此摘录主要为早期单种子开发结果；不能宣称全部筛选无效。
- experience:cluster_coverage：直接删除完整输入状态组造成退化；组号不具有普遍好坏含义。 范围限制：只完成首内部原点；未运行的随机对照和第二原点不能记录为阴性。
- experience:learning_sampling：训练误差或学习进展评分不自动等于未来价值；控制和额外评分成本应分别报告。 范围限制：训练内部开发反馈；控制复用评分，不能声称总成本匹配。
- experience:recent_proxy：近期分布代理首原点改善未迁移到第二原点；可区分输入分布不能证明训练价值。 范围限制：近期代理来自训练历史；未证明未来分布或条件输出不变。
- experience:cadence_overlap：少量总体删除可能几乎清空局部单元；新评分名称可能重复旧失败成员，训练前检查覆盖与重合。 范围限制：局部覆盖及集合重合是实测事实；尚未独立证明退化的唯一机制。
- experience:leverage_temporal：逆概率校正首原点改善，但第二原点退化；保持期望训练目标不能保证Adam或泛化优势。 范围限制：冻结概率边界下的单一适配，不推广到所有杠杆方法。
- experience:importance_weak_signal：重要性采样有小幅三种子同向开发信号，但部分种子收益很弱，成本与长训练确认未完成。 范围限制：正式验证已反复使用；改善主要由一个通道贡献，不声称独立泛化。
- experience:bilevel_control：双层权重两原点略优于控制，但控制贡献了大部分收益；正式确认待完成。 范围限制：解析代理、内部原点和数值实现校验不能替代正式预测泛化。
- experience:measurement_uncertainty：模拟信号长期恒定是可验证现象；缺少独立采集证据时不能判为测量故障或触发删除。 范围限制：经核实故障为0表示证据不足，不保证全部记录健康。
