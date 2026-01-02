# 🚀 Quick Start - LightGBM DLL Setup

## ⏱️ Total Time: 30-60 minutes

---

## 📋 **Option 1: Automated Setup (Recommended)**

### **Single Command:**
1. **Right-click** `SETUP_DLL.bat`
2. Select **"Run as Administrator"**
3. Follow the prompts
4. **Restart computer** when prompted
5. Run `build_lightgbm_dll.ps1` after restart
6. Done!

---

## 📋 **Option 2: Manual Step-by-Step**

### **Step 1: Install Build Tools** (5 min)
```powershell
# Right-click PowerShell → Run as Administrator
.\setup_build_environment.ps1
```

**Installs:**
- ✅ Chocolatey (package manager)
- ✅ CMake (build system)
- ✅ Git (version control)
- ✅ vcpkg (C++ package manager)

---

### **Step 2: Install Visual Studio** (20 min)
```powershell
# Right-click PowerShell → Run as Administrator
.\install_visual_studio.ps1
```

**Installs:**
- ✅ Visual Studio Build Tools 2022
- ✅ MSVC C++ compiler
- ✅ Windows SDK
- ✅ CMake tools

**Size**: ~6 GB download
**Time**: 10-20 minutes

⚠️ **IMPORTANT**: **Restart your computer** after this step!

---

### **Step 3: Build DLL** (30 min)
```powershell
# After restart, run in PowerShell:
.\build_lightgbm_dll.ps1
```

**What it does:**
1. ✅ Installs LightGBM via vcpkg (15-30 min)
2. ✅ Configures CMake project
3. ✅ Compiles C++ DLL
4. ✅ Copies DLL to MT5 Libraries folder
5. ✅ Copies model to MT5 Files folder

**Output:**
- `lightgbm_mt5.dll` → `MQL5\Libraries\`
- `lightgbm_xauusd.txt` → `Terminal\Common\Files\`

---

### **Step 4: Enable DLL in MT5** (1 min)
1. Open MT5
2. Go to: **Tools → Options**
3. Click: **Expert Advisors** tab
4. Check: **☑ Allow DLL imports**
5. Click: **OK**

---

### **Step 5: Compile EA** (1 min)
1. Open **MetaEditor** (F4 in MT5)
2. Open: `mt5_expert_advisor\XAUUSD_NeuralBot_DLL.mq5`
3. Press **F7** (Compile)
4. Check for: **"0 errors, 0 warnings"**

---

### **Step 6: Test in Strategy Tester** (5 min)
1. Open MT5 **Strategy Tester** (Ctrl+R)
2. Select EA: **XAUUSD_NeuralBot_DLL**
3. Symbol: **XAUUSD**
4. Period: **M1**
5. Dates: **2024-01-01** to **2025-01-01**
6. Model: **Every tick based on real ticks**
7. Click: **Start**

**Expected Output:**
```
========================================
XAUUSD Neural Bot v4.0 (LightGBM DLL)
========================================
Loading model from: C:\...\Common\Files\lightgbm_xauusd.txt
✓ Model loaded successfully via DLL
  Features: 26, Classes: 3
✓ All indicators initialized
========================================
Bot initialized successfully!
========================================
```

---

## ✅ Success Indicators

| Step | Success Sign |
|------|--------------|
| Build Tools | Chocolatey, CMake, vcpkg installed |
| Visual Studio | Restart prompt appears |
| Build DLL | "Build Complete!" message |
| MT5 Compile | "0 errors" in MetaEditor |
| Strategy Tester | "Bot initialized successfully!" |

---

## 🐛 Troubleshooting

### ❌ "Not recognized as Administrator"
**Solution**: Right-click script → "Run as Administrator"

### ❌ "CMake not found"
**Solution**: Restart PowerShell or computer

### ❌ "LightGBM installation takes too long"
**Normal**: vcpkg compiles from source, can take 15-30 minutes

### ❌ "DLL not found in MT5"
**Solution**:
1. Check: `[MT5]\MQL5\Libraries\lightgbm_mt5.dll` exists
2. Enable "Allow DLL imports" in MT5 options
3. Restart MT5

### ❌ "Model file not found"
**Solution**:
1. Copy `MT5_XAUBOT\Files\lightgbm_xauusd.txt`
2. To: `%APPDATA%\MetaQuotes\Terminal\Common\Files\`

---

## 📊 What Gets Installed

| Component | Size | Location |
|-----------|------|----------|
| Chocolatey | ~50 MB | `C:\ProgramData\chocolatey\` |
| CMake | ~100 MB | `C:\Program Files\CMake\` |
| vcpkg | ~500 MB | `C:\vcpkg\` |
| Visual Studio Build Tools | ~6 GB | `C:\Program Files (x86)\Microsoft Visual Studio\` |
| LightGBM (via vcpkg) | ~200 MB | `C:\vcpkg\installed\x64-windows\` |
| **Total** | **~7 GB** | |

---

## ⏱️ Time Breakdown

| Step | Time |
|------|------|
| Build tools install | 5 min |
| Visual Studio install | 10-20 min |
| Computer restart | 2 min |
| LightGBM install | 15-30 min |
| DLL build | 2-5 min |
| MT5 setup | 2 min |
| **Total** | **30-60 min** |

---

## 📁 Files Created

```
C:\vcpkg\                                    # Package manager
C:\vcpkg\installed\x64-windows\              # LightGBM library

lightgbm_mt5_dll\
├── build\                                   # Build directory
│   └── bin\Release\lightgbm_mt5.dll        # Compiled DLL
├── lightgbm_wrapper.cpp                     # DLL source
└── CMakeLists.txt                           # Build config

[MT5]\MQL5\Libraries\lightgbm_mt5.dll       # DLL (deployed)
[MT5]\Terminal\Common\Files\lightgbm_xauusd.txt  # Model (deployed)
```

---

## 🎯 Next Steps After Success

1. ✅ Backtest with full 6-year history (2020-2025)
2. ✅ Compare results with Python backtest
3. ✅ Test on demo account
4. ✅ Monitor performance
5. ✅ Go live with conservative risk

---

## 📞 Need Help?

Check these files:
- `DLL_BUILD_INSTRUCTIONS.md` - Detailed build guide
- `MT5_BACKTEST_ERROR_FIX_SUMMARY.md` - Technical explanation
- Build logs in `lightgbm_mt5_dll\build\`

---

**Ready? Right-click `SETUP_DLL.bat` and select "Run as Administrator"!**
