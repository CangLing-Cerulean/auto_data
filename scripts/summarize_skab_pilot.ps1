$ErrorActionPreference = 'Stop'
$stage = Join-Path (Split-Path -Parent $PSScriptRoot) 'research_runs/agent_research/intervention_diagnosis/20261012_skab_pilot'
$feedback = Get-Content -LiteralPath "$stage/common_feedback_node8.json" -Raw -Encoding utf8 | ConvertFrom-Json
$acceptance = Get-Content -LiteralPath "$stage/acceptance.json" -Raw -Encoding utf8 | ConvertFrom-Json
$lock = Get-Content -LiteralPath "$stage/prediction_lock.json" -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($entry in $lock.files) {
    if ((Get-FileHash -LiteralPath "$stage/$($entry.name)" -Algorithm SHA256).Hash.ToLower() -ne $entry.sha256) { throw 'Prediction identity changed' }
}
$methodScores = foreach ($variant in @('strong_prompt','full')) {
    $initial = Get-Content -LiteralPath "$stage/${variant}_prediction_initial.json" -Raw -Encoding utf8 | ConvertFrom-Json
    $items = foreach ($prediction in $initial.predictions) {
        $response = $feedback.responses | Where-Object action -EQ $prediction.action
        foreach ($scope in @('global','target','complement')) {
            $gain = $response.paired_response.$scope.NMSE_gain_pct
            $actual = [Math]::Sign($gain)
            [pscustomobject]@{action=$prediction.action;scope=$scope;predicted=[int]$prediction.$scope;actual=$actual;NMSE_gain_pct=$gain;abstained=($prediction.$scope -eq 0);correct=if ($prediction.$scope -eq 0) {$null} else {$prediction.$scope -eq $actual}}
        }
    }
    $scopes = foreach ($scope in @('global','target','complement','all')) {
        $subset = if ($scope -eq 'all') { @($items) } else { @($items | Where-Object scope -EQ $scope) }
        $nonzero = @($subset | Where-Object { -not $_.abstained })
        $correct = @($nonzero | Where-Object correct).Count
        @{scope=$scope;total=$subset.Count;predicted=$nonzero.Count;correct=$correct;coverage=$nonzero.Count/$subset.Count;accuracy=if ($nonzero.Count) {$correct/$nonzero.Count} else {$null}}
    }
    $harm = @($items | Where-Object { $_.scope -eq 'global' -and $_.actual -lt 0 })
    $nodes = foreach ($node in @(2,4,8)) {
        $decision = Get-Content -LiteralPath "$stage/${variant}_node${node}.json" -Raw -Encoding utf8 | ConvertFrom-Json
        $gain = if ($decision.recommendation -eq 'B0') {0.0} else { ($feedback.responses | Where-Object action -EQ $decision.recommendation).paired_response.global.NMSE_gain_pct }
        @{node=$node;recommendation=$decision.recommendation;exploration_NMSE_gain_pct=$gain}
    }
    @{variant=$variant;direction_items=@($items);scope_scores=@($scopes);harmful_global_actions=$harm.Count;harm_explicitly_predicted=@($harm | Where-Object predicted -EQ -1).Count;harm_false_benefit=@($harm | Where-Object predicted -EQ 1).Count;harm_abstained=@($harm | Where-Object predicted -EQ 0).Count;initial_recommendation=$initial.initial_recommendation;actual_nodes=@($nodes)}
}
$summary = [ordered]@{status='prospective_common_pool_scored';direction_scoring='Unthresholded sign of paired NMSE difference; descriptive, not statistical significance. Abstentions excluded from accuracy, included in coverage denominator. Global harm abstentions and false benefit separately counted.';methods=@($methodScores);responses=$feedback.responses;rule_learned=$false;rule_transfer_validated=$false;independent_sources_in_this_pilot=1;new_completed_trainings=$acceptance.new_trainings;cumulative_completed_trainings=228+$acceptance.new_trainings;training_seconds=$acceptance.training_seconds;test_access=$false;method_superiority_established=$false;feedback_causal_benefit_established=$false}
$summary | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath "$stage/scientific_summary.json" -Encoding utf8
$methodScores | ForEach-Object { [pscustomobject]@{variant=$_.variant;scores=$_.scope_scores;nodes=$_.actual_nodes;harmful=$_.harmful_global_actions;harm_predicted=$_.harm_explicitly_predicted;harm_false_benefit=$_.harm_false_benefit;harm_abstained=$_.harm_abstained} } | ConvertTo-Json -Depth 10
