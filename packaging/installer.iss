; Inno Setup 6 script — per-user install, no administrator rights required.
; Build: iscc /DAppVersion=1.0.0-rc1 /DNumericVersion=1.0.0 /DSourceDir=..\dist\2D2VR180 /DOutputDir=..\release packaging\installer.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef NumericVersion
  #define NumericVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\2D2VR180"
#endif
#ifndef OutputDir
  #define OutputDir "..\release"
#endif

[Setup]
AppId={{6C1C9B7E-2D2A-4B18-9E5B-2D2A18000001}
AppName=2D2VR180
AppVersion={#AppVersion}
VersionInfoVersion={#NumericVersion}.0
AppPublisher=2D2VR180 contributors
DefaultDirName={localappdata}\Programs\2D2VR180
DefaultGroupName=2D2VR180
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=2D2VR180-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\LICENSE
InfoBeforeFile=..\THIRD_PARTY_NOTICES.md
UninstallDisplayIcon={app}\2D2VR180.exe
SetupIconFile=..\assets\icon.ico
MinVersion=10.0.19041

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\2D2VR180"; Filename: "{app}\2D2VR180.exe"
Name: "{group}\2D2VR180 diagnostics"; Filename: "{cmd}"; Parameters: "/k ""{app}\2d2vr180-cli.exe"" doctor"
Name: "{group}\Uninstall 2D2VR180"; Filename: "{uninstallexe}"
Name: "{userdesktop}\2D2VR180"; Filename: "{app}\2D2VR180.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\2D2VR180.exe"; Description: "Launch 2D2VR180"; Flags: nowait postinstall skipifsilent

; User data (models, runtimes, jobs) lives in %LOCALAPPDATA%\2D2VR180 and is
; intentionally kept on uninstall; the app's Models tab can delete it.
