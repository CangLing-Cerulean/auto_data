$ErrorActionPreference = 'Stop'
$taskPilot = 'research_runs/agent_research/phase4/20261009'
if (Test-Path -LiteralPath "$taskPilot/protocol.json") { throw 'Protocol already registered; preserve it' }
$taskSource = 'research_runs/direct_processing/20261009/confirmation_result.json'
$taskRecipePath = 'research_runs/direct_processing/20261009/processing_recipe.json'
$taskRecipe = Get-Content -LiteralPath $taskRecipePath -Raw | ConvertFrom-Json
$taskCase = [ordered]@{
  record_id = 'experience:registered_source_half_20261009'
  dataset_id = 'metropt3_v1'
  dataset_family = 'metropt3'
  record_type = 'registered_development_case'
  evaluation_scope = 'controlled_development_only'
  lesson = ('研究助手已确认固定DLinear的来源变化混合输入。T标准化后，六个通道各保留自身末值并各取一半自身/来源中心变化。source_channel=' + ($taskRecipe.mapping | ConvertTo-Json -Compress) + '；六个改变通道索引=' + ($taskRecipe.changed_channels | ConvertTo-Json -Compress) + '；这些通道self_fraction=0.5，其余可自映射。窗口全保留且均匀。两个原点改善，三个额外种子平均NMSE改善4.2385%，NMAE退化0.0541%；来源扰乱及无来源控制支持额外信息价值。该配方是提供的历史答案，可复用或质疑，不要求采用。')
  limits = '已探索同一数据的研究者指导成果；重现属于经验组合，不是Agent从零发现，不证明新任务泛化或传感器物理故障。'
  status = 'confirmed_scoped_processing_known_prior'
  source = [ordered]@{path=$taskSource; sha256=(Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash.ToLowerInvariant(); excerpt_sha256=(Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash.ToLowerInvariant(); recipe_path=$taskRecipePath; recipe_sha256=(Get-FileHash -LiteralPath $taskRecipePath -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$taskCase | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath "$taskPilot/registered_case.json" -Encoding utf8
$taskOrder = @()
$taskGroups = @(@('full','no_memory','no_feedback','random_reference'), @('no_memory','random_reference','full','no_feedback'), @('no_feedback','full','random_reference','no_memory'))
for ($taskRepeat=1; $taskRepeat -le 3; $taskRepeat++) {
  foreach ($taskGroup in $taskGroups[$taskRepeat-1]) { $taskOrder += [ordered]@{group=$taskGroup; repeat=$taskRepeat} }
}
$taskProtocol = [ordered]@{
  version = 'phase4-controlled-development-v1'
  scope = 'experience_combination_and_feedback_pilot_on_previously_explored_metropt3'
  date = '2026-10-09'
  registered_searches = 12
  repeats_per_group = 3
  groups = @('full','no_memory','no_feedback','random_reference')
  search_order = $taskOrder
  training_seeds = @(20261101,20261102,20261103)
  external_confirmation_seeds = @(20261201,20261202,20261203)
  search_budget = [ordered]@{candidate_proposals=3; coded_diagnostics=4; API_calls=48; API_tokens=400000; wall_seconds=1800; baseline_trainings=1; steps_per_training=3000; samples_per_training=768000}
  confirmation_budget_max_training_jobs_including_search = 17
  model = 'original_DLinear_7920_parameters'
  fixed_parameters = [ordered]@{learning_rate=0.001; batch_size=256; weight_decay=0.0; moving_average_kernel=25}
  effective_processing_gates = [ordered]@{each_extra_seed_NMSE_improves=$true; mean_NMSE_gain_at_least_pct=1.0; mean_NMAE_deterioration_at_most_pct=1.0; matched_control_advantage_at_least_pct=0.25; no_human_strategy_intervention=$true}
  independent_task = $false
  test_access = $false
  historical_trainings = 134
  previous_engineering_trainings = 5
  previous_direct_trainings = 25
  previous_total_trainings = 164
  memory_v2_logical_sha256 = '742b5963c93618ab173d47557d8a7bb65658f103ebe0f3e0c4bbf2619d034726'
  known_case_sha256 = (Get-FileHash -LiteralPath "$taskPilot/registered_case.json" -Algorithm SHA256).Hash.ToLowerInvariant()
  protocol_document_sha256 = (Get-FileHash -LiteralPath "$taskPilot/PROTOCOL.md" -Algorithm SHA256).Hash.ToLowerInvariant()
  implementation_identity = 'bound_after_engineering_checks_before_first_search_in_batch_started_and_each_preregistration'
  output_directory = '/tmp/auto_data_phase4_20261009'
  statistical_agent_superiority = 'not_established_by_three_repeats'
}
$taskProtocol | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath "$taskPilot/protocol.json" -Encoding utf8
'Registered 12 searches before any phase4 API/training call'
