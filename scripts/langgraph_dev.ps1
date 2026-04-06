# LangGraph dev: force UTF-8 so python-dotenv (encoding=None) reads .env as UTF-8 on Windows.
$env:PYTHONUTF8 = "1"
Set-Location $PSScriptRoot\..
& .\.venv\Scripts\langgraph.exe dev @args
