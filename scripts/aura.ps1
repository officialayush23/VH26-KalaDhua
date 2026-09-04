<#
.SYNOPSIS
    Operate the AURA cache from PowerShell.

.DESCRIPTION
    PowerShell aliases `curl` to `Invoke-WebRequest`, which does not accept curl's `-H` or
    `-d` flags, so every curl command written for bash fails here with a confusing binding
    error rather than a helpful one. This module wraps the engine's HTTP surface in real
    cmdlets so that stops mattering.

    Dot-source it once per session:

        . .\scripts\aura.ps1

    Then:

        Get-AuraHealth
        Invoke-AuraReload
        Get-AuraAudit -Limit 20
        Get-AuraFeedback
        Invoke-AuraBench
        Invoke-AuraInvalidate -Tags "row:product:1292"

    Every function honours $env:AURA_URL, defaulting to http://localhost:8080.
#>

$script:AuraUrl = if ($env:AURA_URL) { $env:AURA_URL } else { "http://localhost:8080" }

function Set-AuraUrl {
    <#  Point every command at a different engine, e.g. a Railway deployment. #>
    param([Parameter(Mandatory)][string]$Url)
    $script:AuraUrl = $Url.TrimEnd('/')
    Write-Host "AURA endpoint set to $script:AuraUrl" -ForegroundColor Cyan
}

function Invoke-Aura {
    <#
        The one place that talks HTTP.

        `Invoke-RestMethod` throws on a non-2xx rather than returning the body, which hides
        the engine's own error message — the useful part. This catches that and surfaces
        what the server actually said.
    #>
    param(
        [Parameter(Mandatory)][string]$Path,
        [ValidateSet('Get','Post','Delete')][string]$Method = 'Get',
        $Body = $null
    )
    $uri = "$script:AuraUrl$Path"
    try {
        if ($null -ne $Body) {
            $json = if ($Body -is [string]) { $Body } else { $Body | ConvertTo-Json -Depth 10 -Compress }
            return Invoke-RestMethod -Method $Method -Uri $uri -ContentType 'application/json' -Body $json -TimeoutSec 30
        }
        return Invoke-RestMethod -Method $Method -Uri $uri -TimeoutSec 30
    }
    catch [System.Net.WebException], [Microsoft.PowerShell.Commands.HttpResponseException] {
        $resp = $_.Exception.Response
        if ($null -ne $resp -and $resp.PSObject.Properties.Name -contains 'StatusCode') {
            $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
            $detail = $reader.ReadToEnd()
            Write-Host "engine returned $($resp.StatusCode): $detail" -ForegroundColor Yellow
        } else {
            Write-Host "cannot reach $uri — is the engine running?" -ForegroundColor Red
            Write-Host "  cd engine; cargo run --release -p aura-server -- --real-values" -ForegroundColor DarkGray
        }
        return $null
    }
}

# ----------------------------------------------------------------- health and state

function Get-AuraHealth   { Invoke-Aura -Path '/healthz' }
function Get-AuraStats    { Invoke-Aura -Path '/v1/stats' }
function Get-AuraPolicy   { Invoke-Aura -Path '/v1/policy' }
function Get-AuraCapacity { Invoke-Aura -Path '/v1/capacity' }
function Get-AuraWorkload { Invoke-Aura -Path '/v1/workload' }
function Get-AuraSupabase { Invoke-Aura -Path '/v1/supabase' }

function Get-AuraFeedback {
    <#
        How well the cache's own predictions have held up.

        `calibration_error` is the number worth watching. If the model says 0.70 and reality
        comes back 0.42, it is confidently wrong, and the confidence floor is the only thing
        keeping the cache sane.
    #>
    Invoke-Aura -Path '/v1/feedback'
}

# ----------------------------------------------------------------- the model

function Invoke-AuraReload {
    <#
        Load the bundles in engine/models into the running engine. No restart.

        A rejected bundle is not a crash: the engine logs which features it could not supply
        and keeps serving with whatever was loaded before, because a stale model beats a
        mis-indexed one.
    #>
    param([ValidateSet('file','supabase')][string]$Source = 'file')
    $result = Invoke-Aura -Path '/v1/model/reload' -Method Post -Body @{ source = $Source }
    if ($null -ne $result) {
        $result | ConvertTo-Json -Depth 6
        Write-Host "`nconfirm it took:" -ForegroundColor Cyan
        $policy = Get-AuraPolicy
        if ($null -ne $policy) {
            Write-Host "  predictor = $($policy.predictor)  (want 'gbdt', not 'heuristic')"
        }
    }
    return $result
}

# ----------------------------------------------------------------- explanations

function Get-AuraAudit {
    <# Recent decisions, each as a sentence with the numbers that produced it. #>
    param([int]$Limit = 20, [switch]$Raw)
    $result = Invoke-Aura -Path "/v1/audit?limit=$Limit"
    if ($null -eq $result) { return }
    if ($Raw) { return $result }

    foreach ($e in $result.entries) {
        $colour = switch ($e.severity) {
            'warning' { 'Red' }
            'notice'  { 'Yellow' }
            default   { 'DarkGray' }
        }
        Write-Host ("{0}  {1,-12}" -f $e.at, $e.label) -ForegroundColor $colour -NoNewline
        Write-Host " $($e.subject)" -ForegroundColor White
        Write-Host "            $($e.message)" -ForegroundColor Gray
    }
    Write-Host "`n$($result.count) shown, $($result.suppressed_routine) routine events sampled away" -ForegroundColor DarkGray
}

function Get-AuraExplain {
    <# Why one specific object is, or is not, resident. #>
    param([Parameter(Mandatory)][string]$Key)
    Invoke-Aura -Path "/v1/explain/$([uri]::EscapeDataString($Key))"
}

# ----------------------------------------------------------------- correctness

function Invoke-AuraInvalidate {
    <#
        Drop or mark stale every cached object built from these tags.

        `Hard` removes immediately — for anything where being wrong is unacceptable, such as
        a price. Soft marks stale, so the next reader gets the old value once while a rebuild
        runs behind it, which is far cheaper than a stampede for derived data like a rollup.
    #>
    param(
        [Parameter(Mandatory)][string[]]$Tags,
        [ValidateSet('hard','soft')][string]$Mode = 'hard',
        [string]$Source = 'manual'
    )
    Invoke-Aura -Path '/v1/invalidate' -Method Post -Body @{
        tags = $Tags; mode = $Mode; source = $Source
    }
}

function Invoke-AuraVersionBump {
    <#
        Retire a whole generation of objects without deleting any of them.

        Use this after a model redeploy. Deleting instead would empty a large part of the
        cache at once and send the entire miss stream at the origin — the cache causing the
        outage it exists to prevent.
    #>
    param([Parameter(Mandatory)][string]$Namespace)
    Invoke-Aura -Path '/v1/version/bump' -Method Post -Body @{ namespace = $Namespace }
}

# ----------------------------------------------------------------- benchmark

function Invoke-AuraBench {
    <# Run every policy over one identical request stream and print the table. #>
    param(
        [string]$Scenario = 'expensive_tail',
        [int]$Requests = 80000,
        [long]$CapacityBytes = 134217728,
        [string[]]$Policies = @('lru','lfu','gds','gdsf','tinylfu','s3fifo','sieve','lecar','aura')
    )
    Write-Host "running $($Policies.Count) policies over $Requests requests on '$Scenario'..." -ForegroundColor Cyan
    $run = Invoke-Aura -Path '/v1/bench/run' -Method Post -Body @{
        scenario = $Scenario; policies = $Policies
        capacity_bytes = $CapacityBytes; requests = $Requests
    }
    if ($null -eq $run) { return }

    $report = Invoke-Aura -Path '/v1/bench/latest'
    if ($null -eq $report) { return $run }

    $report.rows |
        Sort-Object total_cost_usd |
        Format-Table @{L='policy';E={$_.policy}},
                     @{L='hit rate';E={'{0:P1}' -f $_.object_hit_rate}},
                     @{L='byte hit';E={'{0:P1}' -f $_.byte_hit_rate}},
                     @{L='cost USD';E={'{0:N2}' -f $_.total_cost_usd}},
                     @{L='backend reqs';E={$_.backend_requests}} -AutoSize

    if ($report.belady_upper_bound) {
        Write-Host ("Belady ceiling: {0:P1} hit rate, `${1:N2}" -f `
            $report.belady_upper_bound.object_hit_rate, $report.belady_upper_bound.total_cost_usd) -ForegroundColor DarkGray
    }
    Write-Host "winner: $($report.winner)" -ForegroundColor Green
    return $report
}

# ----------------------------------------------------------------- traffic

function Start-AuraScenario {
    param(
        [ValidateSet('steady_zipf','mixed_production','flash_crowd','expensive_tail','scan','cost_spike')]
        [string]$Scenario = 'mixed_production',
        [double]$Speed = 1.0
    )
    Invoke-Aura -Path '/v1/sim/start' -Method Post -Body @{ scenario = $Scenario; speed = $Speed }
}

function Stop-AuraScenario { Invoke-Aura -Path '/v1/sim/stop' -Method Post -Body @{} }

function Invoke-AuraAttack {
    <# Inject a disturbance and watch the policy mixture move. #>
    param(
        [ValidateSet('Scan','FlashCrowd','PopularityShift','CostSpike','ExpensiveTail',
                     'HotKeyEmergence','HotKeyDecay','WorkingSetExplosion','MixedChaos')]
        [Parameter(Mandatory)][string]$Attack,
        [int]$DurationSeconds = 30
    )
    Invoke-Aura -Path '/v1/sim/attack' -Method Post -Body @{
        attack = $Attack; duration_s = $DurationSeconds
    }
}

# ----------------------------------------------------------------- overview

function Show-Aura {
    <# One screen: is it up, what is it doing, and is the model any good. #>
    $health = Get-AuraHealth
    if ($null -eq $health) { return }

    Write-Host "`nAURA  $script:AuraUrl" -ForegroundColor Cyan
    Write-Host ("  uptime      {0:N0}s   version {1}" -f $health.uptime_s, $health.version)

    $stats = Get-AuraStats
    if ($null -ne $stats) {
        Write-Host ("  requests    {0:N0}" -f $stats.requests)
    }
    $policy = Get-AuraPolicy
    if ($null -ne $policy) {
        Write-Host ("  predictor   {0}   ml influence {1:P0}" -f $policy.predictor, $policy.ml_influence)
        if ($policy.mixture) {
            $top = $policy.mixture.PSObject.Properties | Sort-Object Value -Descending | Select-Object -First 3
            Write-Host ("  leading     {0}" -f (($top | ForEach-Object { "$($_.Name) $('{0:P0}' -f $_.Value)" }) -join '   '))
        }
    }
    $fb = Get-AuraFeedback
    if ($null -ne $fb) {
        Write-Host ("  calibration predicted {0:P0} vs realised {1:P0}   error {2:P1}" -f `
            $fb.mean_predicted, $fb.mean_realised, [math]::Abs($fb.calibration_error))
        Write-Host ("  judged      {0:N0} decisions settled, {1:N0} still pending" -f $fb.settled, $fb.pending)
    }
    $cap = Get-AuraCapacity
    if ($null -ne $cap) {
        Write-Host ("  capacity    {0:N0} MB, recommending {1:N0} MB   [{2}]" -f `
            ($cap.logical_bytes / 1MB), ($cap.recommended_bytes / 1MB), $cap.decision)
    }
    Write-Host ""
}

Write-Host "AURA commands loaded. Endpoint: $script:AuraUrl" -ForegroundColor Cyan
Write-Host "  Show-Aura   Get-AuraAudit   Get-AuraFeedback   Invoke-AuraReload   Invoke-AuraBench" -ForegroundColor DarkGray
