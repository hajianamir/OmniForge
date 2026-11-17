; ==============================================================================
; OmniForge v2.0.0 — Inno Setup 6 Windows Installer
; Prerequisites: Build onedir executable first (`build_exe.bat --onedir`)
; Compile with:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" omniforge_installer.iss
; ==============================================================================

#define MyAppName "OmniForge"
#define MyAppShortName "OmniForge"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "OmniForge Engineering"
#define MyAppExeName "OmniForge.exe"

[Setup]
AppId={{8F4E2D1A-9B3C-4E77-B562-1C94827E1000}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion} Enterprise Professional
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\OmniForge
DefaultGroupName={#MyAppName}
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist\installer
OutputBaseFilename=OmniForge-{#MyAppVersion}-Enterprise-Setup
SetupIconFile=..\assets\omniforge.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &Desktop shortcut for OmniForge Enterprise Workbench"; GroupDescription: "Additional shortcuts:"
Name: "startmenu"; Description: "Create Start Menu shortcuts (UI + REST API + Diagnostics)"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "..\..\dist\OmniForge\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\.env.example"; DestDir: "{app}"; DestName: ".env"; Flags: onlyifdoesntexist
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: startmenu
Name: "{group}\OmniForge REST API Server"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--mode api --host 127.0.0.1 --api-port 8000"; Tasks: startmenu
Name: "{group}\OmniForge System Self-Check"; Filename: "{cmd}"; Parameters: "/k ""{app}\{#MyAppExeName}"" --mode check"; Tasks: startmenu
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"; Tasks: startmenu
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName} now"; Flags: nowait postinstall skipifsilent
