$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskStage = Join-Path $taskRoot 'research_runs/agent_research/intervention_diagnosis/20261012_phase0'
function Write-PhaseJson($Name, $Value) {
    $taskPath = Join-Path $taskStage $Name
    if (Test-Path -LiteralPath $taskPath) { throw "Refusing to overwrite: $taskPath" }
    [IO.File]::WriteAllText($taskPath,(($Value | ConvertTo-Json -Depth 40).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
}
function Read-PhaseJson($Relative) {
    Get-Content -LiteralPath (Join-Path $taskRoot $Relative) -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable
}
function PhaseHash($Relative) { (Get-FileHash -LiteralPath (Join-Path $taskRoot $Relative) -Algorithm SHA256).Hash.ToLowerInvariant() }
$taskFixed = [ordered]@{model='individual_DLinear';input_length=32;prediction_length=8;steps=3000;batch_size=256;
    optimizer='Adam';learning_rate=0.001;weight_decay=0.0;moving_average_kernel=25;window_stride=1;
    validation_every=500;evaluation_batch_size=2048;checkpoint_selection='lowest_full_D_NMSE_earlier_tie';
    normalization='all_T_rows_mean_population_std_zero_scale_1';exploration_seed=2026101301;
    confirmation_seeds=@(2026101302,2026101303,2026101304)}
$taskTasks = @()
foreach ($taskId in @('metropt3_v1','skab_v1','uci_hydraulic_v1')) {
    $taskManifest = Read-PhaseJson "data/$taskId/manifest.json"
    $taskRole = switch ($taskId) {
        'metropt3_v1' { 'historical_construction_and_engineering_regression' }
        'skab_v1' { 'prospective_method_development_signal_selection' }
        'uci_hydraulic_v1' { 'prospective_source_confirmation_after_method_freeze' }
    }
    $taskBoundary = switch ($taskId) {
        'metropt3_v1' { 'original_partition; increasing_timestamp; adjacent_gap<=30s; 40_records' }
        'skab_v1' { 'partition+source_file; consecutive_source_row; increasing_datetime; adjacent_gap<=30s; 40_records' }
        'uci_hydraulic_v1' { 'partition+cycle_id; consecutive_sample_offset_seconds; 40_records; never_cross_cycle' }
    }
    $taskTasks += [ordered]@{task_id="${taskId}_intervention_32x8_v1";dataset_id=$taskId;role=$taskRole;
        manifest_sha256=PhaseHash "data/$taskId/manifest.json";sensor_columns=$taskManifest.sensor_columns;
        channels=$taskManifest.sensor_columns.Count;parameters=(528*$taskManifest.sensor_columns.Count);
        split_identity=$taskManifest.splits | ForEach-Object { [ordered]@{name=$_.name;file=$_.file;rows=$_.rows;sha256=$_.sha256} };
        window_boundary=$taskBoundary;fixed_training=$taskFixed;test_access='forbidden_except_integrity';
        status='task_definition_frozen_no_new_training';window_count=$null;
        missing='window index generation and runtime validation not executed in local-only scope'}
}
Write-PhaseJson 'tasks.json' ([ordered]@{version='intervention-task-contract-v1';tasks=$taskTasks;
    source_datasets_unchanged=$true;test_distribution_read=$false})

$taskR1Path='research_runs/agent_research/fused_pilot/20261011_sol_high_r1'
$taskR2Path='research_runs/agent_research/autonomous_memory/20261012_sol_high_r2'
$taskR1=Read-PhaseJson "$taskR1Path/summary.json"
$taskR1Pred=Read-PhaseJson "$taskR1Path/decision_initial.json"
$taskR2=Read-PhaseJson "$taskR2Path/confirmation_result.json"
$taskRecords=@()
function Summarize-Region($Parent, $Child, $Elements) {
    $taskDelta=[ordered]@{}; $taskGain=[ordered]@{}
    foreach ($taskMetric in @('NMSE','NMAE')) {
        $taskDelta[$taskMetric]=$Parent[$taskMetric]-$Child[$taskMetric]
        $taskGain[$taskMetric]=if ($Parent[$taskMetric] -eq 0) {$null} else {100*$taskDelta[$taskMetric]/$Parent[$taskMetric]}
    }
    [ordered]@{elements=$Elements;parent=$Parent;child=$Child;delta=$taskDelta;gain_pct=$taskGain;signed_bias=$null}
}
foreach ($taskOrigin in @('O60','O80')) {
    $taskBase=$taskR1.exploration[$taskOrigin].B0
    foreach ($taskAction in @('UP','DOWN','CTRL_UP','CTRL_DOWN')) {
        $taskValue=$taskR1.exploration[$taskOrigin][$taskAction]
        $taskPrediction=($taskR1Pred.decisions | Where-Object task_id -eq $taskOrigin).predictions[$taskAction]
        $taskRecords += [ordered]@{record_id="R1_${taskOrigin}_${taskAction}";source_family='metropt3';context_id='fused_pilot_r1';
            task_id=$taskOrigin;seed=2026101101;parent_id='B0';action_family='loss_weighting_uniform_sampling';action=$taskAction;
            prestate_ref="$taskR1Path/agent_state_before.json";prestate_sha256=PhaseHash "$taskR1Path/agent_state_before.json";
            prospective_prediction=$taskPrediction;prediction_ref=if ($taskPrediction) {"$taskR1Path/decision_initial.json"} else {$null};
            eligibility=if ($taskPrediction) {'historical_prospective_prediction_same_source_only'} else {'historical_control_no_agent_prediction'};
            response=[ordered]@{global=Summarize-Region $taskBase.raw_overall $taskValue.raw_overall ($taskBase.raw_overall.windows*8*15);
                target=Summarize-Region $taskBase.raw_local $taskValue.raw_local ($taskBase.raw_local.windows*8*15);
                complement=[ordered]@{NMSE_gain_pct=$taskValue.other_NMSE_gain_pct;source_definition='original_R1_other_windows';signed_bias=$null}};
            source_ref="$taskR1Path/summary.json";source_sha256=PhaseHash "$taskR1Path/summary.json";
            limitations=@('same_source_overlapping_origins','one_training_seed','not_new_method_evaluation')}
    }
}
foreach ($taskSeedRow in $taskR2.results) {
    foreach ($taskPair in @(@{parent='baseline';child='control';action='historical_source_half'},@{parent='control';child='candidate';action='source_sign_flip'})) {
        $taskParent=$taskSeedRow[$taskPair.parent];$taskChild=$taskSeedRow[$taskPair.child]
        $taskLocalParent=$taskSeedRow["local_$($taskPair.parent)"];$taskLocalChild=$taskSeedRow["local_$($taskPair.child)"]
        $taskRecords += [ordered]@{record_id="R2_$($taskSeedRow.seed)_$($taskPair.action)";source_family='metropt3';context_id='autonomous_memory_r2';
            task_id='metropt3_original_D';seed=$taskSeedRow.seed;parent_id=$taskPair.parent;action_family='input_representation';action=$taskPair.action;
            prestate=$null;prospective_prediction=$null;eligibility='retrospective_only_for_new_signal_learning';
            response=[ordered]@{global=Summarize-Region $taskParent $taskChild ($taskParent.windows*8*15);
                target=Summarize-Region $taskLocalParent $taskLocalChild ($taskLocalParent.windows*8*$taskLocalParent.channels.Count);
                complement=$null};
            source_ref="$taskR2Path/confirmation_result.json";source_sha256=PhaseHash "$taskR2Path/confirmation_result.json";
            limitations=@('aligned_tensor_export_pending','signed_bias_missing','no_new_signal_prestate','same_context_three_seeds_not_three_tasks')}
    }
}
Write-PhaseJson 'historical_response_collection.json' ([ordered]@{version='historical-response-index-v1';records=$taskRecords;
    record_count=$taskRecords.Count;independent_source_count=1;historical_contexts=2;new_training_count=0;
    purpose='engineering fixtures and hypothesis construction; not held-out method evaluation'})
$taskLong=@()
foreach ($taskRecord in $taskRecords) {
    foreach ($taskRegion in @('global','target')) {
        foreach ($taskMetric in @('NMSE','NMAE')) {
            $taskResponse=$taskRecord.response[$taskRegion]
            $taskLong += [pscustomobject][ordered]@{record_id=$taskRecord.record_id;source_family=$taskRecord.source_family;
                context_id=$taskRecord.context_id;parent=$taskRecord.parent_id;action=$taskRecord.action;seed=$taskRecord.seed;
                region=$taskRegion;metric=$taskMetric;elements=$taskResponse.elements;parent_loss=$taskResponse.parent[$taskMetric];
                child_loss=$taskResponse.child[$taskMetric];delta=$taskResponse.delta[$taskMetric];gain_pct=$taskResponse.gain_pct[$taskMetric];
                evidence='existing_aggregate_receipt_not_new_tensor_export'}
        }
    }
}
[IO.File]::WriteAllText((Join-Path $taskStage 'response_long.csv'),(($taskLong | ConvertTo-Csv -NoTypeInformation) -join "`n")+"`n",[Text.UTF8Encoding]::new($false))
Write-PhaseJson 'prototype_input.json' ([ordered]@{facts=[ordered]@{parent_id='source_half';response_visible=$true;
    scope='retrospective_engineering_replay';shared_records=$taskRecords;missing=@('new_source_responses','intermediate_training_dynamics')};history=@()})
Write-PhaseJson 'cost_summary.json' ([ordered]@{new_training_attempts=0;new_training_completed=0;new_model_evaluations=0;
    cumulative_completed_trainings=228;external_API_calls=0;external_API_tokens=0;internal_assistant_cost=$null;
    audit_subagent_count=1;gpu_evaluation_seconds=0;dataset_integrity_receipt='dataset_verification.json';
    counter_reset=$false;local_work_only_after_user_scope_change=$true;
    failed_actions=@(@{action='sandboxed_server_availability';category='network_DNS';training_started=$false},
        @{action='sandboxed_local_verification_junction';category='filesystem_access';training_started=$false},
        @{action='stage0_code_upload';category='automatic_approval_rejection';uploaded=$false;training_started=$false});
    followup='User selected local protocol and prototypes only; no upload retry'})
Write-PhaseJson 'progress.json' ([ordered]@{phase='local_protocol_and_prototypes';status='implemented_pending_static_acceptance_and_sync';
    evidence_floor_complete=$false;training_queue=@();pending=@('Python runtime tests','aligned R2 tensor export','prospective independent-source responses');
    user_scope='local_only';latest_report='REPORT.md';completed_trainings_total=228})
Write-PhaseJson 'protocol.json' ([ordered]@{protocol_id='intervention_diagnosis_phase0_v1';created_utc=[DateTime]::UtcNow.ToString('o');
    scope='local_protocol_and_two_offline_prototypes';formal_experiments_started=$false;external_API_enabled=$false;
    base_llm='internal_gpt-6.1-sol_high';variants=@('strong_prompt','full');information_policy='identical_facts_history_tools';
    observer_training_calls=@(2,4,8);cumulative_budget=$null;primary='parent_increment_and_confirmed_recommendation';
    safeguards=@('test_isolation','no_retrospective_prestate_backfill','elemental_complement','source_disjoint_signal_selection');
    tasks_sha256=PhaseHash 'research_runs/agent_research/intervention_diagnosis/20261012_phase0/tasks.json';
    document_sha256=PhaseHash 'research_runs/agent_research/intervention_diagnosis/20261012_phase0/PROTOCOL.md'})
Write-Output "Prepared $($taskTasks.Count) task contracts and $($taskRecords.Count) historical response records."
