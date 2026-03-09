"""
AI检测器测试脚本
用于验证AI检测器模块是否可以正常加载和运行
"""
import os
import sys
import traceback

def test_ai_detector():
    """测试AI检测器模块是否正常工作"""
    print("开始测试AI检测器组件...")
    
    try:
        print("1. 尝试导入ai_detector模块")
        from ai_detector import detect_ai_generated
        print("   ✅ 导入成功")
        
        # 检查是否有测试图像
        test_image = None
        possible_paths = ["../frontend/images", ".", "../", "temp"]
        
        print("2. 查找测试图像")
        for path in possible_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        test_image = os.path.join(path, file)
                        break
            if test_image:
                break
        
        if not test_image:
            print("   ❌ 未找到测试图像")
            # 使用一个小测试图像
            print("   创建测试图像...")
            from PIL import Image
            import numpy as np
            
            # 创建一个简单的10x10测试图像
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            test_image = "test_image.jpg"
            img.save(test_image)
            print(f"   ✅ 创建测试图像: {test_image}")
        else:
            print(f"   ✅ 找到测试图像: {test_image}")
        
        print("3. 检查函数调用")
        # 尝试调用函数
        try:
            print(f"   调用detect_ai_generated({test_image})...")
            result, analysis = detect_ai_generated(test_image)
            print(f"   ✅ 函数调用成功，返回结果: {result}")
            print(f"   分析结果包含 {len(analysis)} 个字段")
            
            # 打印主要结果
            print("\n检测结果摘要:")
            print(f"AI生成概率: {result:.2f}")
            print(f"结论: {analysis.get('conclusion', '未知')}")
            
            suspicious = analysis.get('suspicious_features', [])
            if suspicious:
                print("\n可疑特征:")
                for feature in suspicious:
                    print(f"- {feature}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 函数调用失败: {e}")
            traceback.print_exc()
            print("\n函数调用堆栈:")
            traceback.print_exc(file=sys.stdout)
            return False
            
    except ImportError as e:
        print(f"   ❌ 导入模块失败: {e}")
        print("\n导入错误堆栈:")
        traceback.print_exc(file=sys.stdout)
        return False
    except Exception as e:
        print(f"   ❌ 测试过程中发生错误: {e}")
        print("\n错误堆栈:")
        traceback.print_exc(file=sys.stdout)
        return False

if __name__ == "__main__":
    success = test_ai_detector()
    if success:
        print("\n✅ AI检测器测试通过")
        sys.exit(0)
    else:
        print("\n❌ AI检测器测试失败")
        sys.exit(1) 