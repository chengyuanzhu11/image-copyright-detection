import os
import sys
import argparse
from logo_model import train_logo_model, test_prediction, load_logo_model
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Logo检测模型训练主函数"""
    parser = argparse.ArgumentParser(description="训练Logo检测模型")
    parser.add_argument("--data_dir", type=str, default="data/train_and_test/train",
                        help="训练数据目录路径")
    parser.add_argument("--epochs", type=int, default=30,
                        help="训练周期数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--test_image", type=str, default=None,
                        help="用于测试的图像路径")
    parser.add_argument("--test_only", action="store_true",
                        help="仅进行测试，不训练模型")
    parser.add_argument("--brand_as_class", action="store_true", default=True,
                        help="使用品牌名称作为类别（适用于多级目录结构）")
    args = parser.parse_args()
    
    # 检查数据目录是否存在
    if not args.test_only and not os.path.exists(args.data_dir):
        logger.error(f"训练数据目录不存在: {args.data_dir}")
        return 1
        
    # 测试模式
    if args.test_only:
        logger.info("仅进行测试模式")
        model = load_logo_model()
        if model is None:
            logger.error("无法加载已训练的模型，请先训练模型")
            return 1
            
        if args.test_image:
            if not os.path.exists(args.test_image):
                logger.error(f"测试图像不存在: {args.test_image}")
                return 1
                
            logger.info(f"测试图像: {args.test_image}")
            result = test_prediction(model, args.test_image)
            logger.info(f"测试结果: {result}")
        else:
            logger.warning("未指定测试图像，请使用 --test_image 参数")
        
        return 0
        
    # 训练模式
    logger.info(f"开始训练模式，数据目录: {args.data_dir}, 周期数: {args.epochs}, 批次大小: {args.batch_size}")
    logger.info(f"使用品牌作为类别: {args.brand_as_class}")
    
    # 训练模型
    model = train_logo_model(
        args.data_dir, 
        args.epochs, 
        args.batch_size, 
        brand_as_class=args.brand_as_class
    )
    
    # 测试模型
    if args.test_image and os.path.exists(args.test_image):
        logger.info(f"使用图像测试模型: {args.test_image}")
        result = test_prediction(model, args.test_image)
        logger.info(f"测试结果: {result}")
    
    logger.info("训练和测试完成")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 