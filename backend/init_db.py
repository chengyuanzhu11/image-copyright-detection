from app import app, db, Logo
import os
import glob
import logging

logger = logging.getLogger(__name__)

def init_database():
    with app.app_context():
        db.create_all()
        
        logo_count = Logo.query.count()
        if logo_count == 0:
            logger.info("数据库为空，初始化一些示例数据...")
            logger.info("数据库结构初始化完成")
        else:
            logger.info(f"数据库中已有{logo_count}个Logo记录")

def test_detection():
    test_image_path = os.path.join('..', 'data', 'train_and_test', 'test')
    
    if not os.path.exists(test_image_path):
        logger.info(f"测试路径不存在: {test_image_path}")
        return
    
    test_images = glob.glob(os.path.join(test_image_path, '**', '*.jpg'), recursive=True)
    if not test_images:
        test_images = glob.glob(os.path.join(test_image_path, '**', '*.png'), recursive=True)
    
    if test_images:
        test_image = test_images[0] 
        logger.info(f"使用测试图像: {test_image}")

        from model import detect_logo_similarity
        result = detect_logo_similarity(test_image)
        logger.info(f"测试结果: {result}")
    else:
        logger.info(f"在测试文件夹中未找到图像: {test_image_path}")

if __name__ == '__main__':
    logger.info("开始初始化数据库...")
    init_database()
    
    logger.info("数据库初始化完成，开始测试检测功能...")
    test_detection()
    logger.info("所有操作完成")  