import os
import sys
import subprocess
import time

def main():
    print("开始导入Logo-2K+数据集...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    import_script = os.path.join(current_dir, 'import_logos.py')
    if not os.path.exists(import_script):
        print(f"错误: 导入脚本不存在: {import_script}")
        return 1
    
    try:
        print("正在运行导入脚本，这可能需要几分钟时间...")
        result = subprocess.run([sys.executable, import_script], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        
        if result.returncode == 0:
            print("导入成功!")
            print(result.stdout)
            return 0
        else:
            print(f"导入失败，错误码: {result.returncode}")
            print("错误信息:")
            print(result.stderr)
            return 1
            
    except Exception as e:
        print(f"运行导入脚本时出错: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 