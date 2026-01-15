# scripts/seed_data.py
import asyncio
import os
import sys

# 确保能导入 app 模块
sys.path.append(os.getcwd())

from sqlmodel import select
from app.core.database import async_session_maker
from app.models.order import User, Order, OrderStatus  # 引入新定义的枚举


async def seed_data():
    async with async_session_maker() as session:
        # 1. 检查用户是否已存在
        # 使用 session.exec (更符合 SQLModel 习惯)
        result = await session.exec(select(User).where(User.username == "test_user"))
        user = result.first()
        
        if not user:
            print("🌱 Creating test user...")
            user = User(
                username="test_user",
                email="test@example.com",
                full_name="张三"
            )
            session.add(user)
            # flush 会将对象推送到数据库缓冲区，从而获取自动生成的 ID，但暂不提交事务
            await session.flush() 

        # 2. 检查并创建 Mock 订单
        result = await session.exec(select(Order).where(Order.user_id == user.id))
        orders = result.all()
        
        if not orders:
            print("📦 Creating mock orders...")
            
            # 订单 1：已发货
            order1 = Order(
                order_sn="SN20240001",
                user_id=user.id,
                # 使用枚举对象而非硬编码字符串
                status=OrderStatus.SHIPPED,
                total_amount=128.50,
                # JSON 结构保持不变
                items=[{"name": "运动内衣", "qty": 1, "price": 128.50}],
                tracking_number="SF123456789",
                shipping_address="上海市浦东新区张江高科技园区"
            )
            
            # 订单 2：待支付
            order2 = Order(
                order_sn="SN20240002",
                user_id=user.id,
                status=OrderStatus.PENDING,
                total_amount=50.00,
                items=[{"name": "全棉袜子", "qty": 5, "price": 10.00}],
                shipping_address="北京市朝阳区三里屯"
            )
            
            session.add_all([order1, order2])
            
        # 最终统一提交事务
        await session.commit()
        print("✅ Seed data completed.")

if __name__ == "__main__":
    asyncio.run(seed_data())