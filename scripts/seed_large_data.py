# scripts/seed_large_data.py
import asyncio
import os
import sys
import random
import uuid
from datetime import datetime, timedelta

sys.path.append(os.getcwd())

from sqlmodel import select, delete
from app.core.database import async_session_maker
# 导入所有涉及到的模型以进行清理
from app.models.order import User, Order, OrderStatus
from app.models.refund import RefundApplication
from app.models.audit import AuditLog
from app.models.message import MessageCard

# --- 配置参数 ---
USER_COUNT = 200
TOTAL_ORDERS = 500

# --- 随机池 ---
PRODUCT_POOL = [
    {"name": "运动T恤", "price": 99.0},
    {"name": "瑜伽裤", "price": 158.0},
    {"name": "跑步鞋", "price": 599.0},
    {"name": "运动水壶", "price": 45.0},
    {"name": "护膝", "price": 88.0},
    {"name": "全棉运动袜", "price": 15.0},
    {"name": "速干衣", "price": 120.0},
    {"name": "筋膜枪", "price": 899.0}
]

ADDRESS_POOL = [
    "上海市浦东新区张江高科技园区", "北京市朝阳区三里屯街道", 
    "广州市天河区珠江新城", "深圳市南山区科技园", 
    "杭州市西湖区文三路", "成都市武侯区软件园"
]

STATUS_POOL = [
    OrderStatus.PENDING, OrderStatus.PAID, 
    OrderStatus.SHIPPED, OrderStatus.DELIVERED, 
    OrderStatus.CANCELLED
]

async def seed_large_data():
    async with async_session_maker() as session:
        print(f"🧹 正在清理旧数据（按外键约束逆序）...")
        
        # 1. 先删除最底层的关联数据（子表）
        await session.exec(delete(AuditLog))
        await session.exec(delete(MessageCard))
        await session.exec(delete(RefundApplication))
        
        # 2. 再删除中间层数据
        await session.exec(delete(Order))
        
        # 3. 最后删除顶层数据（父表）
        await session.exec(delete(User))
        
        await session.commit()
        print("✅ 旧数据清理完毕。")

        # --- 下面的生成逻辑保持不变，但使用 .exec() 顺应新版本建议 ---

        # 1. 创建用户
        print(f"👤 正在生成 {USER_COUNT} 个用户...")
        users = []
        for i in range(1, USER_COUNT + 1):
            user = User(
                username=f"user_{i:03}",
                email=f"user_{i:03}@example.com",
                full_name=f"测试用户_{i:03}"
            )
            users.append(user)
        
        session.add_all(users)
        await session.flush() 

        # 2. 生成订单
        print(f"📦 正在生成 {TOTAL_ORDERS} 个订单...")
        orders = []
        for i in range(1, TOTAL_ORDERS + 1):
            target_user = random.choice(users)
            num_items = random.randint(1, 3)
            order_items = random.sample(PRODUCT_POOL, num_items)
            
            total_amount = 0
            processed_items = []
            for item in order_items:
                qty = random.randint(1, 2)
                processed_items.append({
                    "name": item["name"],
                    "qty": qty,
                    "price": item["price"]
                })
                total_amount += item["price"] * qty
            
            random_days = random.randint(0, 30)
            created_at = datetime.now() - timedelta(days=random_days)
            status = random.choice(STATUS_POOL)
            
            order = Order(
                order_sn=f"SN{created_at.strftime('%Y%m%d')}{i:05}",
                user_id=target_user.id,
                status=status,
                total_amount=round(total_amount, 2),
                items=processed_items,
                tracking_number=f"SF{uuid.uuid4().hex[:10].upper()}" if status in [OrderStatus.SHIPPED, OrderStatus.DELIVERED] else None,
                shipping_address=random.choice(ADDRESS_POOL) + f"{random.randint(1, 999)}号",
                created_at=created_at.replace(tzinfo=None)
            )
            orders.append(order)

        # 分批写入
        batch_size = 100
        for i in range(0, len(orders), batch_size):
            session.add_all(orders[i:i+batch_size])
            await session.commit()
            print(f"   已写入 {i + len(orders[i:i+batch_size])} / {TOTAL_ORDERS} 个订单...")

        print(f"\n🎉 大规模数据初始化成功！")

if __name__ == "__main__":
    confirm = input("⚠️ 该脚本将清空所有业务表数据，是否继续？(y/n): ")
    if confirm.lower() == 'y':
        asyncio.run(seed_large_data())
    else:
        print("操作已取消。")