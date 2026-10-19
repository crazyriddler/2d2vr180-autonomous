; Inno Setup 6 script — per-user install, no administrator rights required.
; Build: iscc /DAppVersion=1.0.0-rc1 /DNumericVersion=1.0.0 /DSourceDir=..\dist\2D2VR /DOutputDir=..\release packaging\installer.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef NumericVersion
  #define NumericVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\2D2VR"
#endif
#ifndef OutputDir
  #define OutputDir "..\release"
#endif

[Setup]
AppId={{6C1C9B7E-2D2A-4B18-9E5B-2D2A18000001}
AppName=2D2VR
AppVersion={#AppVersion}
VersionInfoVersion={#NumericVersion}.0
AppPublisher=2D2VR contributors
DefaultDirName={localappdata}\Programs\2D2VR
DefaultGroupName=2D2VR
; 2.0 renamed the app (2D2VR180 → 2D2VR): same AppId, so it replaces 1.x; install into the new folder
UsePreviousAppDir=no
UsePreviousGroup=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=2D2VR-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\LICENSE
InfoBeforeFile=..\THIRD_PARTY_NOTICES.md
UninstallDisplayIcon={app}\2D2VR.exe
SetupIconFile=..\assets\icon.ico
MinVersion=10.0.19041

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[InstallDelete]
; a clean program folder, and the 1.x program folder and shortcuts (user data is kept and taken over)
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{localappdata}\Programs\2D2VR180"
Type: filesandordirs; Name: "{userprograms}\2D2VR180"
Type: files; Name: "{userdesktop}\2D2VR180.lnk"

[Icons]
Name: "{group}\2D2VR"; Filename: "{app}\2D2VR.exe"
Name: "{group}\2D2VR diagnostics"; Filename: "{cmd}"; Parameters: "/k ""{app}\2d2vr-cli.exe"" doctor"
Name: "{group}\Uninstall 2D2VR"; Filename: "{uninstallexe}"
Name: "{userdesktop}\2D2VR"; Filename: "{app}\2D2VR.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\2D2VR.exe"; Description: "Launch 2D2VR"; Flags: nowait postinstall skipifsilent

; User data (models, runtimes, jobs) lives in %LOCALAPPDATA%\2D2VR (1.x's %LOCALAPPDATA%\2D2VR180 is
; renamed to it on first start) and is intentionally kept on uninstall; the Components page can delete it.
