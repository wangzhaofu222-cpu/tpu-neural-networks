param()
$ErrorActionPreference = 'Stop'
$texBin = 'D:\Tex Live\data\texlive\2026\bin\windows'
$oldPath = $env:Path
$oldLocale = $env:LC_ALL
$oldLang = $env:LANG
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:Path = $texBin + ';' + $env:Path
    $env:LC_ALL = 'C'
    $env:LANG = 'C'
    & "$texBin\latexmk.exe" -xelatex -interaction=nonstopmode -halt-on-error -outdir=build title_page.tex
    if ($LASTEXITCODE -ne 0) { throw 'Title-page compilation failed.' }
    Copy-Item -LiteralPath 'build\title_page.pdf' -Destination 'title_page.pdf'
    & "$texBin\latexmk.exe" -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
    if ($LASTEXITCODE -ne 0) { throw 'Report compilation failed.' }
    Copy-Item -LiteralPath 'build\main.pdf' -Destination 'Lab0_Report.pdf'
} finally {
    $env:Path = $oldPath
    $env:LC_ALL = $oldLocale
    $env:LANG = $oldLang
    Pop-Location
}
