<#
.SYNOPSIS
  Lock (or unlock) webcam exposure, white balance and focus through DirectShow
  (IAMCameraControl / IAMVideoProcAmp). UVC cameras such as the Logitech C930e keep
  these settings until they are changed again, unplugged, or overridden by another
  app (e.g. Logi Tune).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\camera_lock.ps1 status
  powershell -ExecutionPolicy Bypass -File scripts\camera_lock.ps1 lock
  powershell -ExecutionPolicy Bypass -File scripts\camera_lock.ps1 lock -Exposure -6 -WhiteBalance 4500
  powershell -ExecutionPolicy Bypass -File scripts\camera_lock.ps1 auto

.NOTES
  lock: takes the values the camera's auto mode has currently chosen and fixes them
  in manual mode. Exposure is DirectShow log2 seconds (-5 = 1/32 s); it is capped at
  -5 so the camera can still deliver 30 fps. Run it with the lighting you will record in,
  after the camera has been streaming for a few seconds (e.g. right after camera_check).
#>
param(
    [Parameter(Position = 0)][ValidateSet('status', 'lock', 'auto')][string]$Action = 'status',
    [string]$Device = 'Logitech Webcam C930e',
    [Nullable[int]]$Exposure = $null,
    [Nullable[int]]$WhiteBalance = $null,
    [Nullable[int]]$Focus = $null,
    [int]$MaxExposure = -5
)

$src = @"
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

namespace CamLock {
  [ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface ICreateDevEnum {
    [PreserveSig] int CreateClassEnumerator([In] ref Guid clsid, out IEnumMoniker enumMoniker, int flags);
  }
  [ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IPropertyBag {
    [PreserveSig] int Read([MarshalAs(UnmanagedType.LPWStr)] string name, out object val, IntPtr errorLog);
    [PreserveSig] int Write([MarshalAs(UnmanagedType.LPWStr)] string name, ref object val);
  }
  [ComImport, Guid("C6E13370-30AC-11d0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IAMCameraControl {
    [PreserveSig] int GetRange(int prop, out int min, out int max, out int step, out int def, out int flags);
    [PreserveSig] int Set(int prop, int value, int flags);
    [PreserveSig] int Get(int prop, out int value, out int flags);
  }
  [ComImport, Guid("C6E13360-30AC-11d0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  public interface IAMVideoProcAmp {
    [PreserveSig] int GetRange(int prop, out int min, out int max, out int step, out int def, out int flags);
    [PreserveSig] int Set(int prop, int value, int flags);
    [PreserveSig] int Get(int prop, out int value, out int flags);
  }

  public class Ctl {
    public string Name; public bool IsCam; public int Prop;
    public int Min, Max, Step, Def, Caps, Value, Flags; public bool Ok;
    public string Mode { get { return (Flags & 1) != 0 ? "auto" : "manual"; } }
  }

  public class Camera {
    object filter;
    public Camera(string name) {
      Guid clsidEnum = new Guid("62BE5D10-60EB-11d0-BD3B-00A0C911CE86");
      Guid videoCat  = new Guid("860BB310-5D01-11d0-BD3B-00A0C911CE86");
      Guid iidBase   = new Guid("56a86895-0ad4-11ce-b03a-0020af0ba770");
      ICreateDevEnum de = (ICreateDevEnum)Activator.CreateInstance(Type.GetTypeFromCLSID(clsidEnum));
      IEnumMoniker em;
      if (de.CreateClassEnumerator(ref videoCat, out em, 0) != 0 || em == null) throw new Exception("no video devices");
      IMoniker[] m = new IMoniker[1];
      while (em.Next(1, m, IntPtr.Zero) == 0) {
        object bagObj; Guid iidBag = typeof(IPropertyBag).GUID;
        m[0].BindToStorage(null, null, ref iidBag, out bagObj);
        object fn; ((IPropertyBag)bagObj).Read("FriendlyName", out fn, IntPtr.Zero);
        if (fn != null && fn.ToString() == name) {
          m[0].BindToObject(null, null, ref iidBase, out filter);
          return;
        }
      }
      throw new Exception("camera '" + name + "' not found");
    }
    public Ctl Read(string label, bool isCam, int prop) {
      Ctl c = new Ctl(); c.Name = label; c.IsCam = isCam; c.Prop = prop;
      int hr;
      if (isCam) {
        IAMCameraControl cc = (IAMCameraControl)filter;
        hr = cc.GetRange(prop, out c.Min, out c.Max, out c.Step, out c.Def, out c.Caps);
        if (hr == 0) hr = cc.Get(prop, out c.Value, out c.Flags);
      } else {
        IAMVideoProcAmp va = (IAMVideoProcAmp)filter;
        hr = va.GetRange(prop, out c.Min, out c.Max, out c.Step, out c.Def, out c.Caps);
        if (hr == 0) hr = va.Get(prop, out c.Value, out c.Flags);
      }
      c.Ok = hr == 0;
      return c;
    }
    // flags: 1 = auto, 2 = manual
    public int Write(Ctl c, int value, int flags) {
      if (c.IsCam) return ((IAMCameraControl)filter).Set(c.Prop, value, flags);
      return ((IAMVideoProcAmp)filter).Set(c.Prop, value, flags);
    }
  }
}
"@
if (-not ('CamLock.Camera' -as [type])) { Add-Type -TypeDefinition $src -Language CSharp }

$cam = New-Object CamLock.Camera $Device
# CameraControl: Exposure=4, Focus=6.  VideoProcAmp: WhiteBalance=7, Gain=9, BacklightCompensation=8
$defs = @(
    @{ Label = 'Exposure';     Cam = $true;  Prop = 4 },
    @{ Label = 'Focus';        Cam = $true;  Prop = 6 },
    @{ Label = 'WhiteBalance'; Cam = $false; Prop = 7 },
    @{ Label = 'Gain';         Cam = $false; Prop = 9 },
    @{ Label = 'Backlight';    Cam = $false; Prop = 8 }
)
function Show-Controls {
    foreach ($d in $defs) {
        $c = $cam.Read($d.Label, $d.Cam, $d.Prop)
        if ($c.Ok) {
            '{0,-13} value={1,6}  mode={2,-6}  range=[{3}, {4}] step {5}  default {6}' -f $c.Name, $c.Value, $c.Mode, $c.Min, $c.Max, $c.Step, $c.Def
        } else {
            '{0,-13} not supported' -f $d.Label
        }
    }
}

switch ($Action) {
    'status' { Show-Controls }
    'auto' {
        foreach ($d in $defs | Where-Object { $_.Label -in 'Exposure', 'Focus', 'WhiteBalance' }) {
            $c = $cam.Read($d.Label, $d.Cam, $d.Prop)
            if ($c.Ok -and ($c.Caps -band 1)) { [void]$cam.Write($c, $c.Value, 1) }
        }
        'Exposure, focus and white balance set back to auto.'
        Show-Controls
    }
    'lock' {
        $want = @{ Exposure = $Exposure; Focus = $Focus; WhiteBalance = $WhiteBalance }
        foreach ($d in $defs | Where-Object { $_.Label -in 'Exposure', 'Focus', 'WhiteBalance' }) {
            $c = $cam.Read($d.Label, $d.Cam, $d.Prop)
            if (-not $c.Ok) { "$($d.Label): not supported, skipped"; continue }
            if ($d.Label -eq 'Focus' -and $null -eq $want.Focus -and $c.Value -le 0) {
    'Focus: camera reports 0 (infinity / not settled) - left on auto; pass -Focus N to fix it'
    continue
}
$v = if ($null -ne $want[$d.Label]) { [int]$want[$d.Label] } else { $c.Value }
            if ($d.Label -eq 'Exposure' -and $v -gt $MaxExposure) { $v = $MaxExposure }
            $v = [Math]::Max($c.Min, [Math]::Min($c.Max, $v))
            $hr = $cam.Write($c, $v, 2)
            if ($hr -eq 0) { "$($d.Label): locked at $v" } else { "$($d.Label): Set failed, HRESULT 0x{0:X8}" -f $hr }
        }
        Show-Controls
    }
}
