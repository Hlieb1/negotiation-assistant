Set-Location $PSScriptRoot
$jdk21 = "$env:USERPROFILE\.jdks\jbr-21.0.11"
if (Test-Path "$jdk21\bin\java.exe") { $env:JAVA_HOME = $jdk21 }
.\gradlew.bat assembleDebug
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb install -r app\build\outputs\apk\debug\app-debug.apk
& $adb shell am start -n ua.diploma.negotiation/.MainActivity
