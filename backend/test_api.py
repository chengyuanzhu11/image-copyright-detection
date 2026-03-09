import requests
import json
import os
import time
import argparse
from pathlib import Path

# API基础URL
BASE_URL = 'http://localhost:5000'

def test_train_model(dataset_path=None, brand_as_class=True):
    """测试模型训练API"""
    print("测试模型训练API...")
    
    # 获取默认数据集路径
    if dataset_path is None:
        dataset_path = "data/train_and_test/train"
    
    # 发送训练请求
    response = requests.post(
        f"{BASE_URL}/train_model",
        json={
            'dataset_path': dataset_path,
            'epochs': 10,
            'batch_size': 16,
            'brand_as_class': brand_as_class
        }
    )
    
    if response.status_code != 200:
        print(f"训练请求失败: {response.status_code} - {response.text}")
        return False
    
    print(f"训练请求响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    # 轮询模型状态，直到训练完成或超时
    print("\n监控训练状态...")
    start_time = time.time()
    timeout = 600  # 10分钟超时
    
    while time.time() - start_time < timeout:
        try:
            status_response = requests.get(f"{BASE_URL}/model_status")
            if status_response.status_code == 200:
                status = status_response.json()
                training_status = status.get('training_status', {})
                
                # 打印进度
                progress = training_status.get('progress', 0)
                message = training_status.get('message', '')
                print(f"训练进度: {progress}% - {message}")
                
                # 检查训练是否完成
                if not training_status.get('is_training', False) and progress == 100:
                    print("训练成功完成!")
                    return True
                elif not training_status.get('is_training', False) and "失败" in message:
                    print(f"训练失败: {message}")
                    return False
                    
            else:
                print(f"获取状态失败: {status_response.status_code} - {status_response.text}")
                
            # 等待10秒再次检查
            time.sleep(10)
            
        except Exception as e:
            print(f"监控训练状态时出错: {e}")
            time.sleep(10)
    
    print("训练监控超时")
    return False

def test_detect_logo(image_path):
    """测试Logo检测API"""
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return False
        
    print(f"测试Logo检测API，使用图像: {image_path}")
    
    # 发送检测请求
    with open(image_path, 'rb') as img:
        files = {'image': (os.path.basename(image_path), img, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/detect", files=files)
    
    if response.status_code != 200:
        print(f"检测请求失败: {response.status_code} - {response.text}")
        return False
    
    result = response.json()
    print(f"检测结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # 获取详细检测结果
    if 'detection_id' in result:
        try:
            detailed_response = requests.get(
                f"{BASE_URL}/detailed_result", 
                params={"id": result['detection_id']}
            )
            
            if detailed_response.status_code == 200:
                detailed_result = detailed_response.json()
                print(f"\n详细检测结果:")
                
                # 检测方法
                detection_method = detailed_result.get('detection_method', 'unknown')
                print(f"检测方法: {detection_method}")
                
                # 模型预测结果
                model_prediction = detailed_result.get('model_prediction')
                if model_prediction:
                    print(f"模型预测: 类别 = {model_prediction.get('class')}, 置信度 = {model_prediction.get('confidence'):.4f}")
                
                # 特征相似度结果
                matches = detailed_result.get('matches', [])
                if matches:
                    print("\n特征相似度匹配结果 (前5个):")
                    for i, match in enumerate(matches[:5]):
                        print(f"  {i+1}. {match.get('name')} ({match.get('category', 'unknown')}): {match.get('similarity'):.4f}")
                
                # 相似Logo
                similar_logos = detailed_result.get('similar_logos', [])
                if similar_logos:
                    print("\n模型特征相似Logo (前5个):")
                    for i, logo in enumerate(similar_logos[:5]):
                        print(f"  {i+1}. {logo.get('name')}: {logo.get('similarity'):.4f}")
            else:
                print(f"获取详细结果失败: {detailed_response.status_code} - {detailed_response.text}")
        except Exception as e:
            print(f"获取详细结果时出错: {e}")
    
    return True

def test_model_status():
    """测试模型状态API"""
    print("测试模型状态API...")
    
    try:
        response = requests.get(f"{BASE_URL}/model_status")
        if response.status_code == 200:
            status = response.json()
            print(f"模型状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"获取模型状态失败: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"测试模型状态时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="测试Logo检测API")
    parser.add_argument("--train", type=str, help="训练模型，可选指定数据集路径")
    parser.add_argument("--detect", type=str, help="检测图像中的Logo")
    parser.add_argument("--status", action="store_true", help="获取模型状态")
    parser.add_argument("--brand-as-class", action="store_true", default=True,
                        help="使用品牌名称作为类别（适用于多级目录结构）")
    
    args = parser.parse_args()
    
    if not (args.train or args.detect or args.status):
        parser.print_help()
        return
    
    if args.status:
        test_model_status()
        
    if args.train is not None:
        dataset_path = args.train if args.train != "default" else None
        test_train_model(dataset_path, brand_as_class=args.brand_as_class)
        
    if args.detect:
        test_detect_logo(args.detect)

if __name__ == "__main__":
    main() 