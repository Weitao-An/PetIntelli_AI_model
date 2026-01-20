# PowerShell script to initialize Git repository and push to GitHub
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Current directory: $scriptPath" -ForegroundColor Green

# Check if Git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
} else {
    Write-Host "Git repository already exists" -ForegroundColor Green
}

# Add remote repository
$remoteUrl = "git@github.com:Weitao-An/PetIntelli_AI_model.git"
Write-Host "Adding remote repository: $remoteUrl" -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin $remoteUrl

# Add all files (.gitignore will exclude specified files)
Write-Host "Adding files to staging area..." -ForegroundColor Yellow
git add .

# Check if there are files to commit
$status = git status --porcelain
if ($status) {
    Write-Host "Committing changes..." -ForegroundColor Yellow
    git commit -m "Initial commit: AI model service"
    
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    Write-Host "Please ensure you have SSH key set up and access to the repository." -ForegroundColor Cyan
    git branch -M main
    git push -u origin main
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Trying master branch..." -ForegroundColor Yellow
        git branch -M master
        git push -u origin master
    }
} else {
    Write-Host "No files to commit" -ForegroundColor Yellow
}

Write-Host "Done!" -ForegroundColor Green
