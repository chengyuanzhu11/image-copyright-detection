import os
import sys
import numpy as np
from ai_detector import detect_ai_generated

# 添加转换函数
def convert_numpy_types(obj):
    """将numpy类型转换为Python原生类型，解决JSON序列化问题"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# 测试案例
def test_ai_detection():
    # 检查图像文件是否存在
    temp_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    # 从../frontend/images目录查找测试图像或使用当前目录中的图像
    test_image = None
    possible_paths = [
        "../frontend/images",
        ".",
        "../"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_image = os.path.join(path, file)
                    break
        if test_image:
            break
    
    if not test_image:
        print("错误：找不到测试图像文件。请将PNG或JPG图像放在当前目录或frontend/images目录中。")
        return False
    
    print(f"使用测试图像: {test_image}")
    
    try:
        # 调用AI检测函数
        print("开始AI检测...")
        score, analysis = detect_ai_generated(test_image)
        print(f"检测完成，AI生成概率: {score:.4f}")
        
        # 尝试转换结果（测试JSON序列化）
        print("转换JSON...")
        converted = convert_numpy_types({
            'ai_generated_probability': score,
            'analysis': analysis
        })
        
        print("检测和转换成功！")
        print(f"分析结果包含以下键: {list(analysis.keys())}")
        
        # 检查已知问题的键值
        if 'noise_analysis' in analysis:
            noise_data = analysis['noise_analysis']
            print(f"噪声分析: {list(noise_data.keys())}")
            for k, v in noise_data.items():
                print(f"  {k}: {type(v).__name__} = {v}")
        
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ai_detection()
    if success:
        print("\n✅ 测试通过")
        sys.exit(0)
    else:
        print("\n❌ 测试失败")
        sys.exit(1) 