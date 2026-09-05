<#
    Start the whole demo: both example applications on this machine, talking to the
    deployed engine.

    Why local applications against a cloud engine
    ---------------------------------------------
    This is not a compromise, it is the architecture. L1 lives inside each application
    process and removes a network round trip. L2 is the engine, shared by every process,
    and removes a rebuild. Running the applications here and the engine there puts a real
    network between the two tiers, which is the only arrangement in which the difference
    between them is visible rather than asserted.

    It is also the arrangement you want in front of an audience: you can start and stop
    traffic mid-sentence, and neither application sleeps.

    Usage
    -----
        .\scripts\demo.ps1 -RecommendationKey aura_sk_... -AnalyticsKey aura_sk_...

    Mint the two keys in the console's Connect tab. The secret is shown once, at mint time,
    and is not recoverable afterwards -- if you have lost them, mint two more and revoke the
    old ones.
#>

[CmdletBinding()]
param(
    [string] $Engine = "https://vh26-kaladhua.onrender.com",
    [string] $RecommendationKey = $env:AURA_RECOMMENDATION_KEY,
    [string] $AnalyticsKey = $env:AURA_ANALYTICS_KEY,
    [switch] $NoWarm
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$apps = Join-Path $root "apps"
$python = Join-Path $root "aura\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

function Say($text)  { Write-Host $text }
function Good($text) { Write-Host $text -ForegroundColor Green }
function Warn($text) { Write-Host $text -ForegroundColor Yellow }
function Bad($text)  { Write-Host $text -ForegroundColor Red }

Say ""
Say "AURA demo"
Say "---------"
Say "engine   $Engine"
Say "python   $python"
Say ""

if (-not $RecommendationKey -or -not $AnalyticsKey) {
    Bad "Two application keys are required."
    Say ""
    Say "The engine is running enforced, so an application without a key is refused and"
    Say "you would spend the demo looking at an empty dashboard. Mint them here:"
    Say ""
    Say "  1. open the console, sign in, go to Connect"
    Say "  2. mint a key named 'recommendation', copy the secret"
    Say "  3. mint a key named 'analytics', copy the secret"
    Say ""
    Say "then:"
    Say "  .\scripts\demo.ps1 -RecommendationKey aura_sk_xxx -AnalyticsKey aura_sk_yyy"
    Say ""
    exit 1
}

# The free instance sleeps, and the first request after that pays about fifty seconds of
# cold start. Paying it here, before anything is watching, is the difference between a demo
# that opens on a dashboard and one that opens on a spinner.
if (-not $NoWarm) {
    Say "Waking the engine (a sleeping free instance takes up to a minute) ..."
    $awake = $false
    for ($i = 0; $i -lt 24; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "$Engine/healthz" -TimeoutSec 10 -UseBasicParsing
            if ($r.StatusCode -eq 200) { $awake = $true; break }
        } catch {
            Start-Sleep -Seconds 5
        }
    }
    if ($awake) { Good "  engine is awake" } else { Warn "  engine did not answer; starting the applications anyway" }
    Say ""
}

function Start-App($name, $module, $key, $port) {
    $env:AURA_APPS_AURA_BASE_URL = $Engine
    $env:AURA_API_KEY = $key
    $env:PORT = "$port"
    Say "starting $name on port $port"
    # A separate window per service, so its log is readable and Ctrl+C in one does not take
    # the other down with it.
    $args = @(
        "-NoExit", "-Command",
        "`$env:AURA_APPS_AURA_BASE_URL='$Engine'; `$env:AURA_API_KEY='$key'; `$env:PORT='$port'; " +
        "Set-Location '$apps'; & '$python' -m $module"
    )
    Start-Process -FilePath "powershell.exe" -ArgumentList $args -WindowStyle Normal | Out-Null
}

Start-App "recommendation" "recommendation.main" $RecommendationKey 8101
Start-App "analytics"      "analytics.main"      $AnalyticsKey      8102

Say ""
Say "waiting for both to answer ..."
$ready = @{}
foreach ($svc in @(@{n="recommendation"; p=8101}, @{n="analytics"; p=8102})) {
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:$($svc.p)/health" -TimeoutSec 3 -UseBasicParsing
            if ($r.StatusCode -eq 200) { $ready[$svc.n] = $true; break }
        } catch {
            Start-Sleep -Milliseconds 700
        }
    }
    if ($ready[$svc.n]) { Good "  $($svc.n) is up" } else { Bad "  $($svc.n) did not come up -- check its window" }
}

Say ""
Say "Open these"
Say "----------"
Say "  storefront (recommendation)  http://localhost:8101/"
Say "  storefront (analytics)       http://localhost:8102/"
Say "  traffic control panel        http://localhost:8101/control"
Say "  console                      https://universe-ten-iota.vercel.app"
Say "  engine                       $Engine"
Say ""
Say "The order that tells the story"
Say "------------------------------"
Say "  1. console, Connect tab: both applications now show as connected"
Say "  2. a storefront: click a few products, watch served_from go origin then l1"
Say "  3. control panel: start 4000 users at 40 req/s"
Say "  4. console, Evidence tab: the six charts fill as the traffic runs"
Say "  5. control panel: flash crowd, then price change, then model redeploy"
Say ""
