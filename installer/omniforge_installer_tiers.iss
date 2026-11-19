; ==============================================================================
; OmniForge v2.0.0 — Tiered / Component-Based Windows Installer (Inno Setup 6)
;
; Small installer + on-demand provisioning:
;   * payload  = app code + embeddable CPython + uv.exe + locks/*.lock   (~55-75 MB)
;   * wizard   = Simple / Advanced / Everything, plus per-pack checkboxes
;   * install  = provision.py bootstraps a private venv and installs ONLY the
;                selected packs (reusing local wheels first, downloading the rest)
;
; Two launch flavours are supported (choose with /DLaunchMode=...):
;   LaunchMode=py    (DEFAULT)  no PyInstaller step; the app is started through the bundled
;                                embeddable CPython:  OmniForge.exe  =  shim -> runtime\python
;   LaunchMode=exe              ships a PyInstaller build (build_exe.bat --onedir) as
;                                dist\OmniForge\OmniForge.exe and installs that instead
;
; Compile (default, recommended):
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\omniforge_installer_tiers.iss
; Compile with a PyInstaller launcher:
;   ISCC.exe /DLaunchMode=exe installer\omniforge_installer_tiers.iss
; Offline:  pass /WHEELHOUSE="\\srv\wheels" (or get it from the answer file) to install with no internet.
; ==============================================================================

#define MyAppName "OmniForge"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "OmniForge Engineering"
#define MyAppExeName "OmniForge.exe"

; Staging directory produced by scripts\build_bootstrap.py.
; Override from the command line if you stage elsewhere:
;   ISCC.exe /DStagingDir=C:\build\omniforge-bootstrap installer\omniforge_installer_tiers.iss
#ifndef StagingDir
  #define StagingDir "..\dist\omniforge-bootstrap"
#endif

#ifndef LaunchMode
  #define LaunchMode "py"
#endif

; ------------------------------------------------------------------------------
; Fail fast with an actionable message instead of ISCC's "cannot find the path".
; NOTE: ISPP has no AddBackslash(); #ifnexist takes a plain (relative or absolute) path.
; ------------------------------------------------------------------------------
; Guard the file ISCC actually reads: [Setup] LicenseFile needs .txt/.rtf, so the payload
; carries LICENSE.txt (a twin of LICENSE). Guarding the wrong twin is how "Line 73: Could not
; read ...app\LICENSE.txt" slips past a guard that still passes on app\LICENSE.
#ifnexist StagingDir + "\app\LICENSE.txt"
  #error "Bootstrap payload is missing or stale (no app\LICENSE.txt). Re-run build_installer.bat, or: python scripts\build_bootstrap.py --launcher-zip python-3.13.9-amd64.zip"
#endif
#ifnexist StagingDir + "\app\omniforge.py"
  #error "Payload is incomplete: app\omniforge.py not found. Re-run scripts\build_bootstrap.py."
#endif
#ifnexist StagingDir + "\runtime\python\python.exe"
  #error "Payload is incomplete: runtime\python\python.exe not found. Re-run scripts\build_bootstrap.py."
#endif
#ifnexist StagingDir + "\locks\advanced.lock"
  #error "Payload is incomplete: locks\advanced.lock not found. Run scripts\build_locks.py, then scripts\build_bootstrap.py."
#endif

; Where the user-facing executable ends up, per launch mode.
;   py  -> the shim ships inside the runtime folder (next to python3XX.dll)
;   exe -> the PyInstaller build sits at the app root
#if LaunchMode == "exe"
  #define MyAppExePath "{app}\" + MyAppExeName
#else
  #define MyAppExePath "{app}\runtime\python\" + MyAppExeName
#endif

[Setup]
AppId={{8F4E2D1A-9B3C-4E77-B562-1C94827E1000}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion} Enterprise
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\OmniForge
DefaultGroupName={#MyAppName}
LicenseFile={#StagingDir}\app\LICENSE.txt
OutputDir=..\dist\installer
OutputBaseFilename=OmniForge-{#MyAppVersion}-Setup
SetupIconFile={#StagingDir}\app\packaging\assets\omniforge.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={#MyAppExePath}
DisableProgramGroupPage=yes
; The provisioning step downloads from PyPI; the uninstaller must not delete user data.
UninstallDisplayName={#MyAppName} {#MyAppVersion}
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

; ------------------------------------------------------------------------------
; Tier selection (radio buttons on the "Select Components" page)
; ------------------------------------------------------------------------------
[Types]
Name: "simple";     Description: "Simple  -  Studio 01-05 + diagnostics (fallback backends, ~0.8 GB)"
Name: "advanced";   Description: "Advanced  -  all core studios incl. the API/workbench server (~0.9 GB)"
Name: "everything"; Description: "Everything  -  Advanced + every native backend (PyTorch, XGBoost, CV, NLP...)"

; ------------------------------------------------------------------------------
; Packs. `Types:` decides which tier preselects the pack; `Flags: unchecked`
; means optional packs stay off unless the user ticks them.
; ------------------------------------------------------------------------------
[Components]
Name: "core_runtime"; Description: "Core runtime  (numpy, pandas, scipy, scikit-learn, pydantic...)"; Types: simple advanced everything; Flags: fixed
Name: "statistics";   Description: "Forecasting & advanced statistics  (statsmodels)";                    Types: simple advanced everything
Name: "data_io";      Description: "Fast data connectors  (Parquet, DuckDB, SQL, XML)";                    Types: simple advanced everything
Name: "reporting";    Description: "Executive reporting  (PDF, DOCX, PPTX, XLSX, IPYNB)";                  Types: simple advanced everything
Name: "api_server";   Description: "Workbench server  (FastAPI + uvicorn + JWT/RBAC)";                     Types: advanced everything

Name: "boosting";     Description: "Gradient boosting  (XGBoost / LightGBM / CatBoost, ~260 MB)";        Types: everything; Flags: unchecked
Name: "deep";         Description: "Deep learning  (PyTorch, ~2.8 GB installed)";                        Types: everything; Flags: unchecked
Name: "vision";       Description: "Computer vision  (OpenCV + scikit-image, ~180 MB)";                   Types: everything; Flags: unchecked
Name: "nlp";          Description: "NLP toolkit  (NLTK / spaCy / gensim, ~220 MB)";                      Types: everything; Flags: unchecked
Name: "foundation";   Description: "Foundation models  (transformers, CLIP, YOLO, ~1.5 GB)";             Types: everything; Flags: unchecked
Name: "automl";       Description: "Extra AutoML backends  (Optuna / FLAML, ~40 MB)";                     Types: everything; Flags: unchecked
Name: "responsible";  Description: "Explainability & carbon  (SHAP + codecarbon, ~90 MB)";              Types: everything; Flags: unchecked

[Tasks]
Name: "desktopicon"; Description: "Create a &Desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: checkedonce
Name: "startmenu";   Description: "Create Start Menu shortcuts"; GroupDescription: "Additional shortcuts:"; Flags: checkedonce

; ------------------------------------------------------------------------------
; Payload: application code + embeddable Python + uv + lock files. NOT the wheels.
; ------------------------------------------------------------------------------
[Files]
; Application payload staged by scripts/build_bootstrap.py
Source: "{#StagingDir}\app\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

#if LaunchMode == "exe"
; PyInstaller desktop build (produced by build_exe.bat --onedir)
Source: "..\dist\OmniForge\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif
; LaunchMode=py needs no extra payload: the shim lives in runtime\python (staged already) so the
; Windows loader finds python3XX.dll next to it.
; Embeddable CPython 3.13 (pinned: the lock files target cp313)
Source: "{#StagingDir}\runtime\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs
; Hash-pinned lock files (one per tier)
Source: "{#StagingDir}\locks\*.lock"; DestDir: "{app}\locks"; Flags: ignoreversion
; Provisioning script + default config.
; NOTE: app\* above already copies .env.example and app\installer\provision.py, so only the
; rename to ".env" is needed here (a second copy of the same file would be a duplicate entry).
Source: "{#StagingDir}\app\.env.example"; DestDir: "{app}"; DestName: ".env"; Flags: onlyifdoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{#MyAppExePath}"; IconFilename: "{app}\packaging\assets\omniforge.ico"; Tasks: startmenu
Name: "{group}\{#MyAppName} REST API"; Filename: "{#MyAppExePath}"; Parameters: "--mode api"; IconFilename: "{app}\packaging\assets\omniforge.ico"; Tasks: startmenu
Name: "{group}\Repair {#MyAppName} installation"; Filename: "{app}\runtime\python\python.exe"; Parameters: """{app}\installer\provision.py"" --repair --tier {code:GetTierArgument} --packs ""{code:GetPackList}"" --app-dir ""{app}"""; Tasks: startmenu
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"; Tasks: startmenu
Name: "{autodesktop}\{#MyAppName}"; Filename: "{#MyAppExePath}"; IconFilename: "{app}\packaging\assets\omniforge.ico"; Tasks: desktopicon

; ------------------------------------------------------------------------------
; Provisioning runs after the files land; the installer window stays responsive.
; `--wheelhouse` (optional) enables a fully offline install from a share.
; ------------------------------------------------------------------------------
[Run]
Filename: "{app}\runtime\python\python.exe"; \
  Parameters: """{app}\installer\provision.py"" --tier {code:GetTierArgument} --packs ""{code:GetPackList}"" --app-dir ""{app}"" --status ""{tmp}\omniforge_provision.json"" {code:GetExtraArgs}"; \
  StatusMsg: "Installing the selected OmniForge capability packs (this may download from PyPI)..."; \
  Flags: waituntilterminated runhidden; \
  Check: NeedProvisioning

; Both launch modes point at {#MyAppExePath}; in py mode the shim starts the venv-backed app.
Filename: "{#MyAppExePath}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\runtime\python\python.exe"; Parameters: "-c ""import shutil,os;shutil.rmtree(os.path.join(os.environ.get('LOCALAPPDATA',''),'OmniForge','venv'),ignore_errors=True)"""; Flags: runhidden

[Code]
var
  WheelhouseEdit: TEdit;
  WheelhousePage: TInputDirWizardPage;

{ The tier comes from the wizard's type selection; optional packs are additive on top of it
  (provision.py accepts --tier advanced --packs deep). }
function GetTierArgument(Param: String): String;
begin
  if IsComponentSelected('api_server') then
    Result := 'advanced'
  else
    Result := 'simple';
end;

{ Optional packs are passed individually so the provisioner installs exactly what
  was ticked (never more). }
function GetPackList(Param: String): String;
var
  Packs: String;
begin
  Packs := '';
  if IsComponentSelected('boosting')    then Packs := Packs + 'boosting,';
  if IsComponentSelected('deep')        then Packs := Packs + 'deep,';
  if IsComponentSelected('vision')      then Packs := Packs + 'vision,';
  if IsComponentSelected('nlp')         then Packs := Packs + 'nlp,';
  if IsComponentSelected('foundation')  then Packs := Packs + 'foundation,';
  if IsComponentSelected('automl')      then Packs := Packs + 'automl,';
  if IsComponentSelected('responsible') then Packs := Packs + 'responsible,';
  if Length(Packs) > 0 then
    SetLength(Packs, Length(Packs) - 1);
  Result := Packs;
end;

function GetExtraArgs(Param: String): String;
begin
  Result := '';
  if WheelhouseEdit <> nil then
    if Trim(WheelhouseEdit.Text) <> '' then
      Result := '--wheels "' + Trim(WheelhouseEdit.Text) + '"';
end;

{ Provisioning runs for interactive AND silent installs (an unattended install must still be
  usable). For a files-only copy, compile with /DNoProvision=1. }
function NeedProvisioning: Boolean;
begin
#ifdef NoProvision
  Result := False;
#else
  Result := True;
#endif
end;

procedure InitializeWizard;
begin
  WheelhousePage := CreateInputDirPage(wpSelectComponents,
    'Optional: offline wheelhouse',
    'Where are pre-downloaded wheels located?',
    'Leave empty to download from PyPI (internet required). Point this at a local folder or a ' +
    'network share (\\server\wheels) for a fully offline installation.',
    False, '');
  WheelhousePage.Add('');
  WheelhouseEdit := WheelhousePage.Edits[0];
end;

{ Refuse to continue when the user picked an online install but nothing can reach PyPI.
  (A full connectivity probe lives in provision.py; this is the fast pre-flight.) }
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (WheelhousePage <> nil) and (CurPageID = WheelhousePage.ID) then begin
    if (Trim(WheelhouseEdit.Text) = '') then
      if MsgBox('No wheelhouse was selected, so the selected packs will be downloaded from PyPI and ' +
                'an internet connection is required during installation.' + #13#10#13#10 +
                'Continue?', mbConfirmation, MB_YESNO) = IDNO then
        Result := False;
  end;
end;
