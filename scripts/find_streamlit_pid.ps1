$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
foreach ($p in $procs) {
  if ($p.CommandLine -like '*streamlit*8599*') { Write-Output $p.ProcessId }
}