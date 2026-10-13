param(
    [string]$PredictionPath = 'research_runs/agent_research/intervention_diagnosis/20261013_hydraulic_signal_r1/predictions.json',
    [string]$OutputPath = 'research_runs/agent_research/intervention_diagnosis/20261013_paper_feasibility_r1/mean_shift_information_audit.json'
)
$ErrorActionPreference = 'Stop'
$taskPred = Get-Content -LiteralPath $PredictionPath -Raw -Encoding utf8 | ConvertFrom-Json
$taskRows = [ordered]@{}
foreach ($taskKey in @('H1_P1','H1_CP1')) {
    $taskSignal = [double]$taskPred.signals.$taskKey
    $taskEdit = [double]$taskPred.input_trace.$taskKey.features.edit_ratio
    $taskError = [double]$taskPred.anchor_metrics.NMSE
    if ($taskSignal -lt 0 -or $taskSignal -ge 1 -or $taskEdit -le 0 -or $taskEdit -ge 1 -or $taskError -le 0) {
        throw 'Energy inversion requires finite interior ratios and positive parent error'
    }
    $taskMeanDose = $taskError*$taskSignal/(1-$taskSignal)
    $taskTotalDose = $taskError*$taskEdit/(1-$taskEdit)
    if (-not [double]::IsFinite($taskMeanDose) -or -not [double]::IsFinite($taskTotalDose) -or $taskMeanDose -gt $taskTotalDose) {
        throw 'Energy decomposition invariant failed'
    }
    $taskRows[$taskKey] = [ordered]@{
        mean_shift_energy=$taskMeanDose; total_edit_energy=$taskTotalDose;
        centered_edit_energy=$taskTotalDose-$taskMeanDose;
        mean_component_share=$taskMeanDose/$taskTotalDose;
        centered_component_share=1-$taskMeanDose/$taskTotalDose;
        edited_fraction=$taskPred.input_trace.$taskKey.features.edited_fraction;
        source='locked training-input summary'; posthoc=$true
    }
}
[ordered]@{
    status='PASS'; type='algebraic_reconstruction_of_preexisting_statistics'; cells=$taskRows;
    prediction_source_sha256=(Get-FileHash -LiteralPath $PredictionPath).Hash.ToLowerInvariant();
    new_trainings=0; new_predictions=0; new_thresholds=0; test_access=$false;
    interpretation_limit='Describes information omitted by the mean-shift signal, not proof that omitted energy caused response or that adding it predicts retraining gains.'
} | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Output "Energy decomposition saved: $OutputPath"
