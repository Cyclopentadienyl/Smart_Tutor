import os
import sys
import subprocess
import venv
from pathlib import Path


def setup_environment():
    """
    自動化部署環境：
    1. 檢查 Python 版本
    2. 建立虛擬環境 (.venv)
    3. 升級 pip
    4. 安裝 requirements.txt
    """
    print("=== 開始自動部署 AI 家教系統環境 ===")

    # 1. 檢查 Python 版本
    if sys.version_info < (3, 10):
        print("❌ 錯誤：需要 Python 3.10 或以上版本。")
        return

    venv_dir = Path(".venv")

    # 2. 建立虛擬環境
    if not venv_dir.exists():
        print(f"🔨 正在建立虛擬環境於 {venv_dir} ...")
        venv.create(venv_dir, with_pip=True)
    else:
        print(f"✅ 虛擬環境已存在：{venv_dir}")

    # 決定 pip 的路徑 (Windows vs Unix)
    if os.name == 'nt':  # Windows
        python_executable = venv_dir / "Scripts" / "python.exe"
        pip_executable = venv_dir / "Scripts" / "pip.exe"
    else:  # Mac/Linux
        python_executable = venv_dir / "bin" / "python"
        pip_executable = venv_dir / "bin" / "pip"

    # 3. 安裝套件
    print("📦 正在安裝/更新 依賴套件 (這可能需要幾分鐘)...")
    try:
        # 升級 pip
        subprocess.check_call([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"])

        # 安裝 requirements.txt
        subprocess.check_call([str(pip_executable), "install", "-r", "requirements.txt"])

        print("\n✅ 環境部署成功！")
        print("========================================")
        print("請使用以下指令啟動系統：")
        if os.name == 'nt':
            print(f".venv\\Scripts\\python.exe launch.py")
        else:
            print(f".venv/bin/python launch.py")
        print("========================================")

    except subprocess.CalledProcessError as e:
        print(f"❌ 安裝過程中發生錯誤：{e}")


if __name__ == "__main__":
    setup_environment()