#!/usr/bin/env python3
# test/test_users.py
"""
初始化测试用户和订单数据
用于横向越权测试
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import async_session_maker, init_db
from app.models.user import User
from app.models.order import Order, OrderStatus
from sqlmodel import select


async def init_test_data():
    """初始化测试数据"""
    print("=" * 60)
    print("🚀 初始化测试用户和订单数据")
    print("=" * 60)
    
    await init_db()
    
    async with async_session_maker() as session:
        # 检查是否已存在测试用户
        result = await session.exec(
            select(User).where(User.username == "alice")
        )
        if result.first():
            print("⚠️  测试用户已存在，跳过初始化")
            return
        
        # 创建测试用户
        users_data = [
            {
                "username": "alice",
                "password": "alice123",
                "email": "alice@example.com",
                "full_name": "Alice Wang",
                "phone": "13800138001",
                "is_admin":  False
            },
            {
                "username": "bob",
                "password": "bob123",
                "email": "bob@example.com",
                "full_name": "Bob Li",
                "phone": "13800138002",
                "is_admin": False
            },
            {
                "username": "admin",
                "password": "admin123",
                "email": "admin@example.com",
                "full_name": "Admin User",
                "phone": "13800138000",
                "is_admin":  True
            }
        ]
        
        users = []
        for user_data in users_data:
            password = user_data.pop("password")
            user = User(
                **user_data,
                password_hash=User.hash_password(password)
            )
            session.add(user)
            users.append(user)
        
        await session.commit()
        
        # 刷新以获取 ID
        for user in users:
            await session.refresh(user)
        
        print(f"\n✅ 创建了 {len(users)} 个测试用户:")
        for user in users: 
            print(f"   - {user.username} (ID: {user.id}, Admin: {user.is_admin})")
        
        # 为 Alice 创建订单
        alice = users[0]
        bob = users[1]
        
        alice_orders = [
            Order(
                order_sn="SN20240001",
                user_id=alice.id,
                status=OrderStatus.DELIVERED,
                total_amount=299.00,
                items=[
                    {"name": "无线鼠标", "qty": 1, "price": 99.00},
                    {"name": "机械键盘", "qty": 1, "price": 200.00}
                ],
                shipping_address="北京市朝阳区 xxx 小区 1-101",
                tracking_number="SF1234567890"
            ),
            Order(
                order_sn="SN20240002",
                user_id=alice.id,
                status=OrderStatus.SHIPPED,
                total_amount=1599.00,
                items=[
                    {"name": "降噪耳机", "qty":  1, "price": 1599.00}
                ],
                shipping_address="北京市朝阳区 xxx 小区 1-101",
                tracking_number="SF1234567891"
            ),
            Order(
                order_sn="SN20240003",
                user_id=alice.id,
                status=OrderStatus.DELIVERED,
                total_amount=2599.00,
                items=[
                    {"name": "智能手表", "qty": 1, "price": 2599.00}
                ],
                shipping_address="北京市朝阳区 xxx 小区 1-101",
                tracking_number="SF1234567892"
            )
        ]
        
        # 为 Bob 创建订单
        bob_orders = [
            Order(
                order_sn="SN20240004",
                user_id=bob.id,
                status=OrderStatus.PAID,
                total_amount=3999.00,
                items=[
                    {"name": "平板电脑", "qty":  1, "price": 3999.00}
                ],
                shipping_address="上海市浦东新区 yyy 大厦 5-202"
            ),
            Order(
                order_sn="SN20240005",
                user_id=bob.id,
                status=OrderStatus.DELIVERED,
                total_amount=599.00,
                items=[
                    {"name": "蓝牙音箱", "qty": 1, "price": 599.00}
                ],
                shipping_address="上海市浦东新区 yyy 大厦 5-202",
                tracking_number="SF1234567893"
            )
        ]
        
        all_orders = alice_orders + bob_orders
        for order in all_orders:
            session.add(order)
        
        await session.commit()
        
        print(f"\n✅ 创建了 {len(all_orders)} 个测试订单:")
        print(f"   - Alice 的订单: {len(alice_orders)} 个")
        for order in alice_orders:
            print(f"     • {order.order_sn}:  ¥{order.total_amount} ({order.status})")
        print(f"   - Bob 的订单: {len(bob_orders)} 个")
        for order in bob_orders:
            print(f"     • {order.order_sn}: ¥{order.total_amount} ({order.status})")
    
    print("\n" + "=" * 60)
    print("🎉 测试数据初始化完成！")
    print("=" * 60)
    print("\n📝 测试账号:")
    print("   Alice: alice / alice123 (普通用户)")
    print("   Bob:    bob / bob123 (普通用户)")
    print("   Admin: admin / admin123 (管理员)")
    print("\n🔐 横向越权测试场景:")
    print("   1.使用 Bob 登录，尝试查询 Alice 的订单 SN20240001")
    print("   2.预期结果:  系统拒绝访问")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(init_test_data())