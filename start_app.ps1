$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
python -m streamlit run ".\pharma_agent\ui\app.py" --server.runOnSave false --server.fileWatcherType none
