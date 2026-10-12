$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pilot = Join-Path $root 'research_runs/agent_research/intervention_diagnosis/20261012_skab_pilot'
$output = Join-Path $root 'research_runs/agent_research/intervention_diagnosis/20261012_stage1_readiness'
$UnchangedBandPercent = 0.25
$feedback = Get-Content -LiteralPath "$pilot/common_feedback_node8.json" -Raw -Encoding utf8 | ConvertFrom-Json
$methods = foreach ($variant in @('strong_prompt','full')) {
    $initial = Get-Content -LiteralPath "$pilot/${variant}_prediction_initial.json" -Raw -Encoding utf8 | ConvertFrom-Json
    $items = foreach ($prediction in $initial.predictions) {
        $response = $feedback.responses | Where-Object action -EQ $prediction.action
        foreach ($scope in @('global','target','complement')) {
            $gain = [double]$response.paired_response.$scope.NMSE_gain_pct
            $observed = if ($gain -gt $UnchangedBandPercent) {'improve'} elseif ($gain -lt -$UnchangedBandPercent) {'harm'} else {'unchanged'}
            $declared = if ($prediction.$scope -gt 0) {'improve'} elseif ($prediction.$scope -lt 0) {'harm'} else {'uncertain'}
            [pscustomobject]@{action=$prediction.action;scope=$scope;predicted_raw_direction=$declared;gain_pct=$gain;observed_with_band=$observed;review=if ($declared -eq 'uncertain') {'abstained'} elseif ($observed -eq $declared) {'direction_supported_beyond_band'} elseif ($observed -eq 'unchanged') {'inside_unchanged_band'} else {'opposite_beyond_band'}}
        }
    }
    [pscustomobject]@{variant=$variant;items=@($items);declared_direction_count=@($items | Where-Object predicted_raw_direction -NE 'uncertain').Count;supported_beyond_band=@($items | Where-Object review -EQ 'direction_supported_beyond_band').Count;inside_unchanged_band=@($items | Where-Object review -EQ 'inside_unchanged_band').Count;opposite_beyond_band=@($items | Where-Object review -EQ 'opposite_beyond_band').Count;abstentions=@($items | Where-Object review -EQ 'abstained').Count}
}
if ($methods[0].supported_beyond_band -ne 2 -or $methods[0].inside_unchanged_band -ne 2 -or $methods[1].supported_beyond_band -ne 3 -or $methods[1].inside_unchanged_band -ne 4) {throw 'Manual cross-check differs'}
foreach ($method in $methods) { if ($method.declared_direction_count+$method.abstentions -ne 24 -or $method.supported_beyond_band+$method.inside_unchanged_band+$method.opposite_beyond_band -ne $method.declared_direction_count) {throw 'Denominator mismatch'} }
$inputPaths = @('research_runs/agent_research/intervention_diagnosis/20261012_phase0/PROTOCOL.md','online_ts/agent_research/intervention.py','research_runs/agent_research/intervention_diagnosis/20261012_skab_pilot/PROTOCOL.md','research_runs/agent_research/intervention_diagnosis/20261012_skab_pilot/common_feedback_node8.json','research_runs/agent_research/intervention_diagnosis/20261012_skab_pilot/full_prediction_initial.json','research_runs/agent_research/intervention_diagnosis/20261012_skab_pilot/strong_prompt_prediction_initial.json')
$inputs = foreach ($path in $inputPaths) { @{path=$path;sha256=(Get-FileHash -LiteralPath (Join-Path $root $path) -Algorithm SHA256).Hash.ToLower()} }
$audit = [ordered]@{status='additive_retrospective_scoring_clarification';created_utc=[DateTime]::UtcNow.ToString('o');unchanged_band_pct=$UnchangedBandPercent;reason='Phase0 dual-update contract defines unchanged band; SKAB pilot descriptive scorer used raw sign. Preserve raw-sign results; do not reinterpret these additional counts as a new preregistered benchmark.';methods=@($methods);all_8_global_responses_inside_band=@($feedback.responses | Where-Object { [Math]::Abs($_.paired_response.global.NMSE_gain_pct) -le $UnchangedBandPercent }).Count -eq 8;S_UP_target_review='Raw sign differs from initial improve prediction, but observed magnitude lies inside unchanged band. It is not evidence of material opposite response.';recipe_gate_unchanged=$true;final_recommendations_unchanged=@{full='B0';strong_prompt='B0'};inputs=@($inputs);new_trainings=0;new_model_evaluations=0;test_access=$false;old_protocols_predictions_scores_modified=$false}
$audit | ConvertTo-Json -Depth 25 | Set-Content -LiteralPath "$output/scoring_audit.json" -Encoding utf8
$methods | Select-Object variant,declared_direction_count,supported_beyond_band,inside_unchanged_band,opposite_beyond_band,abstentions | ConvertTo-Json
