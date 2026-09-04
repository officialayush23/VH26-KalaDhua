<#
.SYNOPSIS
    Operate the AURA cache from PowerShell.

.DESCRIPTION
    PowerShell aliases `curl` to `Invoke-WebRequest`, which does not accept curl's `-H` or
    `-d` flags, so every curl command written for bash fails here with a confusing binding
    error rather than a helpful one. This module wraps the engine's HTTP surface in real
    cmdlets so that stops mattering.

    This file is deliberately pure ASCII and saved with a byte-order mark. Windows
    PowerShell 5.1 decodes a .ps1 as ANSI unless it finds a BOM, so a single typographic
    dash becomes three bytes that include a quote character, which terminates a string
    mid-function and produces a "Missing closing '}'" error pointing at the wrong line.

    Dot-source it once per session:

        . .\scripts\aura.ps1

    Then:

        Show-Aura
        Invoke-AuraBench
        Get-AuraAudit -Limit 20
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
        the engine's own error message - the useful part. This catches that and surfaces
        what the server actually said.

        The catch is untyped on purpose: HttpResponseException does not exist in Windows
        PowerShell 5.1, and naming a type the runtime cannot resolve fails the whole
        function rather than the one call.
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
            return Invoke-RestMethod -Method $Method -Uri $uri -ContentType 'application/json' -Body $json -TimeoutSec 300
        }
        return Invoke-RestMethod -Method $Method -Uri $uri -TimeoutSec 300
    }
    catch {
        # PowerShell 7 puts the body in ErrorDetails; 5.1 makes you read the stream.
        $detail = $null
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
            $detail = $_.ErrorDetails.Message
        }
        $resp = $_.Exception.Response
        if ($null -eq $detail -and $null -ne $resp -and $resp.PSObject.Properties.Name -contains 'GetResponseStream') {
            try {
                $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())
                $detail = $reader.ReadToEnd()
            } catch { }
        }
        if ($detail) {
            Write-Host "engine said: $detail" -ForegroundColor Yellow
        } else {
            Write-Host "cannot reach $uri - is the engine running?" -ForegroundColor Red
            Write-Host "  cd engine; cargo run --release -p aura-server -- --real-values" -ForegroundColor DarkGray
        }
        return $null
    }
}

# ----------------------------------------------------------------- health and state

function Get-AuraHealth      { Invoke-Aura -Path '/healthz' }
function Get-AuraStats       { Invoke-Aura -Path '/v1/stats' }
function Get-AuraPolicy      { Invoke-Aura -Path '/v1/policy' }
function Get-AuraCapacity    { Invoke-Aura -Path '/v1/capacity' }
function Get-AuraWorkload    { Invoke-Aura -Path '/v1/workload' }
function Get-AuraSupabase    { Invoke-Aura -Path '/v1/supabase' }
function Get-AuraConsistency { Invoke-Aura -Path '/v1/consistency' }

function Get-AuraFeedback {
    <#
        How well the cache's own predictions have held up.

        `calibration_error` is the number worth watching. If the model says 0.70 and reality
        comes back 0.42, it is confidently wrong, and the confidence floor is the only thing
        keeping the cache sane.
    #>
    Invoke-Aura -Path '/v1/feedback'
}

function Get-AuraRefreshQueue {
    <# What the cache owes a rebuild on, with enough context for an application to do it. #>
    param([int]$Limit = 20)
    Invoke-Aura -Path "/v1/refresh/queue?limit=$Limit"
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
    param([int]$Limit = 20, [string]$Kind, [switch]$Raw)
    $result = Invoke-Aura -Path "/v1/audit?limit=$Limit"
    if ($null -eq $result) { return }
    if ($Raw) { return $result }

    $entries = $result.entries
    if ($Kind) { $entries = $entries | Where-Object { $_.kind -eq $Kind } }

    foreach ($e in $entries) {
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

        `hard` removes immediately - for anything where being wrong is unacceptable, such as
        a price. `soft` marks stale, so the next reader gets the old value once while a
        rebuild runs behind it, which is far cheaper than a stampede for derived data like a
        rollup.
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
        cache at once and send the entire miss stream at the origin - the cache causing the
        outage it exists to prevent.
    #>
    param([Parameter(Mandatory)][string]$Namespace)
    Invoke-Aura -Path '/v1/version/bump' -Method Post -Body @{ namespace = $Namespace }
}

# ----------------------------------------------------------------- benchmark

function Invoke-AuraBench {
    <#
        Run every policy over one identical request stream and print the table.

        Each policy is its real implementation, not a scoring function standing in for it,
        and each is charged the metadata it actually carries.
    #>
    param(
        [string]$Scenario = 'expensive_tail',
        [int]$Requests = 80000,
        [long]$CapacityBytes = 134217728,
        [string[]]$Policies = @('lru','fifo','lfu','gds','gdsf','tinylfu','s3fifo','sieve','lecar','aura')
    )
    Write-Host "running $($Policies.Count) policies over $Requests requests on '$Scenario'..." -ForegroundColor Cyan
    $report = Invoke-Aura -Path '/v1/bench/run' -Method Post -Body @{
        scenario = $Scenario; policies = $Policies
        capacity_bytes = $CapacityBytes; requests = $Requests
    }
    if ($null -eq $report) { return }

    $report.rows |
        Sort-Object total_cost_usd |
        Format-Table @{L='policy';E={$_.policy}},
                     @{L='hit rate';E={'{0:P1}' -f $_.object_hit_rate}},
                     @{L='byte hit';E={'{0:P1}' -f $_.byte_hit_rate}},
                     @{L='cost USD';E={'{0:N4}' -f $_.total_cost_usd}},
                     @{L='backend';E={$_.backend_requests}},
                     @{L='mean MB';E={'{0:N0}' -f ($_.mean_resident_bytes / 1MB)}},
                     @{L='meta KB';E={'{0:N0}' -f ($_.memory_overhead_bytes / 1KB)}} -AutoSize

    if ($report.belady_upper_bound) {
        Write-Host ("Belady ceiling: {0:P1} hit rate at `${1:N4}" -f `
            $report.belady_upper_bound.object_hit_rate, $report.belady_upper_bound.total_cost_usd) -ForegroundColor DarkGray
    }
    Write-Host "winner: $($report.winner)" -ForegroundColor Green
    Write-Host "aura vs each baseline (positive = aura is cheaper):" -ForegroundColor Cyan
    $report.improvement_vs.PSObject.Properties |
        Sort-Object { -[double]$_.Value } |
        ForEach-Object { Write-Host ("  {0,-9} {1,7:P2}" -f $_.Name, [double]$_.Value) }
    return $report
}

function Invoke-AuraBenchSuite {
    <#
        The five scenarios the brief cares about, in one go.

        The claim we have to defend is "beats conventional caching on at least three
        scenarios", so running one and quoting it is not evidence.
    #>
    param([int]$Requests = 80000, [long]$CapacityBytes = 134217728)
    $summary = @()
    foreach ($s in @('expensive_tail','mixed_production','flash_crowd','cost_spike','scan')) {
        Write-Host "`n===== $s =====" -ForegroundColor Magenta
        $r = Invoke-AuraBench -Scenario $s -Requests $Requests -CapacityBytes $CapacityBytes
        if ($null -eq $r) { continue }
        $aura = $r.rows | Where-Object { $_.policy -eq 'aura' }
        $best = $r.rows | Where-Object { $_.policy -ne 'aura' } | Sort-Object total_cost_usd | Select-Object -First 1
        $summary += [pscustomobject]@{
            scenario      = $s
            winner        = $r.winner
            aura_usd      = $aura.total_cost_usd
            best_rival    = $best.policy
            rival_usd     = $best.total_cost_usd
            margin        = if ($best.total_cost_usd -gt 0) { ($best.total_cost_usd - $aura.total_cost_usd) / $best.total_cost_usd } else { 0 }
        }
    }
    Write-Host "`n===== summary =====" -ForegroundColor Magenta
    $summary | Format-Table @{L='scenario';E={$_.scenario}},
                            @{L='winner';E={$_.winner}},
                            @{L='aura USD';E={'{0:N4}' -f $_.aura_usd}},
                            @{L='best rival';E={$_.best_rival}},
                            @{L='rival USD';E={'{0:N4}' -f $_.rival_usd}},
                            @{L='margin';E={'{0:P2}' -f $_.margin}} -AutoSize
    $won = ($summary | Where-Object { $_.winner -eq 'aura' }).Count
    $colour = if ($won -ge 3) { 'Green' } else { 'Red' }
    Write-Host "aura wins $won of $($summary.Count) scenarios (the brief asks for 3)" -ForegroundColor $colour
    return $summary
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
        Write-Host ("  requests    {0:N0}" -f $stats.engine.admissions)
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
    $c = Get-AuraConsistency
    if ($null -ne $c) {
        Write-Host ("  correctness {0:N0} tags tracked, {1:N0} keys invalidated, {2:N0} served stale" -f `
            $c.tracked_tags, $c.keys_invalidated, $c.stale_serves)
        Write-Host ("  singleflight {0:N0} origin calls suppressed, {1:N0} rebuilds owed" -f `
            $c.single_flight.origin_calls_suppressed, $c.refresh_backlog)
    }
    $cap = Get-AuraCapacity
    if ($null -ne $cap) {
        Write-Host ("  capacity    {0:N0} MB, recommending {1:N0} MB   [{2}]" -f `
            ($cap.logical_bytes / 1MB), ($cap.recommended_bytes / 1MB), $cap.decision)
    }
    Write-Host ""
}

Write-Host "AURA commands loaded. Endpoint: $script:AuraUrl" -ForegroundColor Cyan
Write-Host "  Show-Aura   Invoke-AuraBenchSuite   Get-AuraAudit   Get-AuraConsistency   Invoke-AuraReload" -ForegroundColor DarkGray
