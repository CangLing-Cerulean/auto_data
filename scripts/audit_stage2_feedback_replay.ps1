param([string]$Stage='research_runs/agent_research/intervention_diagnosis/20261013_stage2_feedback_replay_r1')
$ErrorActionPreference='Stop'
function Read-Json([string]$Path) { Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable }
function Hash-File([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Require([bool]$Value,[string]$Message) { if (-not $Value) { throw $Message } }
function Same-Json($Left,$Right) { ($Left | ConvertTo-Json -Depth 60 -Compress) -ceq ($Right | ConvertTo-Json -Depth 60 -Compress) }
function Save-New([string]$Path,$Value) {
    Require (-not (Test-Path -LiteralPath $Path)) "Output exists: $Path"
    [IO.File]::WriteAllText($Path,($Value | ConvertTo-Json -Depth 60)+"`n",[Text.UTF8Encoding]::new($false))
}
$plan=Read-Json "$Stage/protocol.json"
$preflight=Read-Json "$Stage/preflight.json"
$oracle=Read-Json "$Stage/oracle.json"
$result=Read-Json "$Stage/result.json"
Require ($plan.script_sha256 -eq (Hash-File 'scripts/stage2_feedback_replay.py')) 'Frozen script changed'
foreach ($name in $plan.local_files_sha256.Keys) { Require ($plan.local_files_sha256[$name] -eq (Hash-File "$Stage/$name")) "Frozen file changed: $name" }
Require ($preflight.protocol_sha256 -eq (Hash-File "$Stage/protocol.json")) 'Protocol hash mismatch'
Require ($preflight.common_facts_sha256 -eq (Hash-File "$Stage/common_facts.json")) 'Common facts mismatch'
Require ($preflight.oracle_sha256 -eq (Hash-File "$Stage/oracle.json")) 'Oracle mismatch'
$count=2
$details=[ordered]@{}
foreach ($arm in $plan.arms) {
    $initial=Read-Json "$Stage/initial_$($arm.Substring(0,2))_output.json"
    $completion=Read-Json "$Stage/arms/$arm/completion.json"
    for ($step=1; $step -le 3; $step++) {
        $inputPath="$Stage/arms/$arm/step${step}_input.json"
        $outputPath="$Stage/arms/$arm/step${step}_output.json"
        $packet=Read-Json $inputPath
        $decision=Read-Json $outputPath
        Require ($decision.packet_id -eq $packet.packet_id) 'Packet identity mismatch'
        Require ($completion.output_sha256[[string]$step] -eq (Hash-File $outputPath)) 'Completed output hash mismatch'
        Require ($packet.common_facts_sha256 -eq $preflight.common_facts_sha256) 'Common fact identity mismatch'
        if ($step -eq 1) { Require (Same-Json $packet.common_prefix_initial_judgment $initial) 'Shared initial judgment differs' }
        else { Require ($packet.previous_output_sha256 -eq (Hash-File "$Stage/arms/$arm/step$($step-1)_output.json")) 'Previous decision hash mismatch' }
        Require (($arm[1] -eq '1') -eq $packet.ContainsKey('learned_signal')) 'Signal mask mismatch'
        Require (($arm[3] -eq '1') -eq $packet.ContainsKey('separate_updates')) 'Dual update mask mismatch'
        foreach ($path in $decision.files_read) { Require ($path -in $packet.allowed_files) 'Declared file access outside whitelist' }
        foreach ($feedback in $packet.feedback) {
            if ($arm.EndsWith('M')) {
                Require (-not $feedback.ContainsKey('response')) 'Masked effect leaked'
                Require ($packet.separate_updates[$feedback.id].recipe -eq 'unknown') 'Masked recipe state leaked'
                Require ($packet.separate_updates[$feedback.id].hypothesis -eq 'unknown') 'Masked hypothesis state leaked'
            } else {
                $expectedScopes=if ($arm.EndsWith('G')) { @('global') } else { @('global','target','complement') }
                Require ((($feedback.response.Keys | Sort-Object) -join ',') -eq (($expectedScopes | Sort-Object) -join ',')) 'Scope mask differs'
                foreach ($scope in $expectedScopes) {
                    Require (Same-Json $feedback.response[$scope].parent $oracle.initial[$feedback.id].parent_metrics[$scope]) 'Parent response differs'
                    Require (Same-Json $feedback.response[$scope].child $oracle.initial[$feedback.id].child_metrics[$scope]) 'Child response differs'
                    Require (Same-Json $feedback.response[$scope].gain_percent $oracle.initial[$feedback.id].gain_percent[$scope]) 'Response gain differs'
                }
            }
        }
        $count++
    }
    foreach ($pair in @(@(2,2),@(4,3))) {
        $node=[string]$pair[0]
        $packet=Read-Json "$Stage/arms/$arm/step$($pair[1])_input.json"
        $decision=Read-Json "$Stage/arms/$arm/step$($pair[1])_output.json"
        $score=$result.outcomes[$arm]["node$node"]
        $raw=0.;$qualified=0.;$violations=0;$harm=0;$unobserved=0
        foreach ($task in @('S1','S2')) {
            $action=$decision.recommendations[$task]
            if ($action -eq 'B0') { $gain=0.;$eligible=$true }
            else {
                $cell=$oracle.confirmation["${task}_${action}"]
                $gain=($cell.repeats | ForEach-Object { $_.gain_percent.global.NMSE } | Measure-Object -Average).Average
                $eligible=$cell.confirmed_adoption
                if ("${task}_${action}" -notin $packet.observed_ids) { $unobserved++ }
            }
            $raw+=$gain
            if ($eligible) { $qualified+=$gain } else { $violations++ }
            if ($gain -lt -.25) { $harm++ }
        }
        Require ([Math]::Abs($score.raw_mean_NMSE_gain-$raw/2) -lt 1e-12) 'Raw score mismatch'
        Require ([Math]::Abs($score.qualified_mean_NMSE_gain-$qualified/2) -lt 1e-12) 'Qualified score mismatch'
        Require ($score.gate_violations -eq $violations -and $score.harmful_choices -eq $harm -and $score.unobserved_recommendations -eq $unobserved) 'Count mismatch'
    }
}
Require ($count -eq 20 -and $result.LLM_calls -eq 20) 'Decision count mismatch'
$full=$result.outcomes.S1U1F.node2
$advance=$true
foreach ($other in @('S0U1F','S1U0F','S1U1G','S1U1M')) {
    $comparator=$result.outcomes[$other].node2
    $passed=$full.qualified_mean_NMSE_gain -gt $comparator.qualified_mean_NMSE_gain -and $full.gate_violations -le $comparator.gate_violations -and $full.harmful_choices -le $comparator.harmful_choices
    Require ($passed -eq $result.incremental_comparisons[$other].passed) 'Comparison gate mismatch'
    $advance=$advance -and $passed
}
Require ($advance -eq $result.advance_to_new_prospective_design) 'Advancement mismatch'
foreach ($key in $oracle.confirmation.Keys) {
    $cell=$oracle.confirmation[$key]
    $details[$key]=[ordered]@{confirmed_adoption=$cell.confirmed_adoption;mean_NMSE_gain_percent=$cell.mean_global_NMSE_gain_percent;repeats=@($cell.repeats | ForEach-Object { [ordered]@{seed=$_.seed;global_NMSE_gain_percent=$_.gain_percent.global.NMSE;global_NMAE_gain_percent=$_.gain_percent.global.NMAE} })}
}
Save-New "$Stage/confirmation_scoring_details.json" $details
Save-New "$Stage/final_audit.json" ([ordered]@{status='PASS';checked_utc=[DateTime]::UtcNow.ToString('o');decision_outputs=$count;arms=6;scoring_nodes=12;frozen_inputs_verified=$true;shared_prefix_verified=$true;feedback_mask_verified=$true;recommendation_scores_recomputed=$true;decision_hash_chain_verified=$true;access_validation='Declared files_read checked against per-packet whitelist; complete independent tool-access trace unavailable; no OS sandbox claim';advance_to_new_prospective_design=$advance;new_trainings=0})
Write-Output "PASS: $count decisions, 6 branches, 12 node scores; advancement=$advance"
