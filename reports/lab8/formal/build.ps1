param()
$ErrorActionPreference = 'Stop'
$oldPath = $env:Path
$oldLocale = $env:LC_ALL
$oldLang = $env:LANG
Push-Location -LiteralPath $PSScriptRoot
try {
    $latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
    if (-not $latexmk) {
        $localTex = 'D:\Tex Live\data\texlive\2026\bin\windows'
        if (Test-Path -LiteralPath "$localTex\latexmk.exe") {
            $env:Path = $localTex + ';' + $env:Path
            $latexmk = Get-Command latexmk -ErrorAction Stop
        } else { throw 'Install TeX Live with XeLaTeX and latexmk, and add its bin directory to PATH.' }
    }
    $env:LC_ALL = 'C'
    $env:LANG = 'C'
    & $latexmk.Source -xelatex -interaction=nonstopmode -halt-on-error -outdir=build title_page.tex
    if ($LASTEXITCODE -ne 0) { throw 'Title-page compilation failed.' }
    Copy-Item -LiteralPath 'build\title_page.pdf' -Destination 'title_page.pdf'
    & $latexmk.Source -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
    if ($LASTEXITCODE -ne 0) { throw 'Report compilation failed.' }
    Copy-Item -LiteralPath 'build\main.pdf' -Destination 'Lab8_Report.pdf'
} finally {
    $env:Path = $oldPath
    $env:LC_ALL = $oldLocale
    $env:LANG = $oldLang
    Pop-Location
}
