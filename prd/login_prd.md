**登录注册功能细化设计**

## 1. 用户表结构（PostgreSQL）
CREATE TABLE users (
    id VARCHAR(50) PRIMARY KEY,                 -- 业务主键（UUID或雪花ID）
    username VARCHAR(100) UNIQUE NOT NULL,       -- 登录账号（唯一，不可为空）
    password_hash VARCHAR(255) NOT NULL,         -- 密码哈希值（bcrypt）
    nickname VARCHAR(100),                        -- 昵称（可选）
    avatar_url TEXT,                              -- 头像URL（可选）
    created_at BIGINT NOT NULL,                    -- 创建时间戳（毫秒）
    updated_at BIGINT NOT NULL,                     -- 更新时间戳（毫秒）
    is_deleted BOOLEAN DEFAULT FALSE
);

-- 索引

CREATE INDEX idx_users_username ON users(username) WHERE is_deleted = FALSE;
### 说明

username 作为登录唯一标识，可以是任意字符串（如邮箱、自定义用户名）。

password_hash 使用bcrypt算法加密存储，不可逆。

其他字段为可选信息，可在注册时一并提供或后续完善。

## 2. 用户认证方式
### 2.1 注册
流程

用户填写用户名、密码（及可选昵称），点击注册。
前端调用 /api/auth/register，传递用户名、密码等。
后端校验用户名是否已存在，若不存在则对密码进行哈希加密，创建用户记录。
生成JWT返回给前端，前端保存token。
### 2.2 登录
流程

用户填写用户名和密码，点击登录。
前端调用 /api/auth/login。
后端根据用户名查找用户，比对密码哈希值。若匹配，生成JWT返回；否则返回错误。
前端保存token。
### JWT设计

载荷包含 user_id、username、过期时间（建议7天）。

密钥存储在服务端环境变量，使用HS256算法。
## 3. 未登录检测与提示逻辑
3.1 前端实现
进入页面：不强制检查登录，直接展示对话界面（可显示默认内容）。

发送消息前检测：
当用户输入问题并点击发送时，执行以下逻辑：
async function sendMessage() {
  if (!isLoggedIn()) {  // 检查本地是否有token
    showLoginDialog();  // 弹出登录提示框
    return;
  }
  // 已登录，正常发送消息
  await doSend();
}
登录提示框：
底部弹出半屏面板（或模态框），包含提示文字“请先登录或注册”，以及两个按钮：“取消”和“去登录”。点击“去登录”跳转至登录页面（或打开登录弹窗）。

登录页面：
包含用户名输入框、密码输入框、登录按钮、注册按钮（或切换标签）。
登录成功后，自动执行待发送的消息（需在弹出登录前保存用户输入的消息内容到变量 pendingMessage）。
已登录态维持：
应用启动时（onLaunch/onLoad）读取本地token，并调用 /api/user/info 验证有效性。若失效则清除token。
3.2 后端需提供接口
GET /api/user/info：验证token有效性，返回用户基本信息。

所有需要登录的接口（如会话列表、发送消息）需校验JWT。

## 4. 后端API详细设计
4.1 注册
URL: POST /api/auth/register

请求体:

json
{
  "username": "john_doe",
  "password": "securePassword123",
  "nickname": "约翰"   // 可选
}
响应（成功）:

json
{
  "code": 200,
  "message": "注册成功",
  "data": {
    "token": "eyJhbGc...",
    "user": {
      "id": "xxx",
      "username": "john_doe",
      "nickname": "约翰",
      "avatar": null
    }
  }
}
错误:

400: 用户名已存在
400: 用户名或密码格式不正确（如长度要求，可自行定义）

4.2 登录
URL: POST /api/auth/login

请求体:

json
{
  "username": "john_doe",
  "password": "securePassword123"
}
响应:

json
{
  "code": 200,
  "message": "登录成功",
  "data": {
    "token": "eyJhbGc...",
    "user": {
      "id": "xxx",
      "username": "john_doe",
      "nickname": "约翰",
      "avatar": null
    }
  }
}
错误:

401: 用户名或密码错误

4.3 获取用户信息
URL: GET /api/user/info

请求头: Authorization: Bearer <token>

响应:

json
{
  "code": 200,
  "message": "success",
  "data": {
    "id": "xxx",
    "username": "john_doe",
    "nickname": "约翰",
    "avatar": "https://...",
    "created_at": 1620000000000
  }
}
错误:

401: token无效或过期

4.4 其他需登录的接口（示例）
创建会话、发送消息等接口均需在请求头携带token，后端验证用户身份。

## 5. 数据库关联调整
现有 sessions 表中的 user_id 需与 users.id 关联。建议添加外键约束以保证数据一致性（可选但推荐）：

sql
ALTER TABLE sessions 
ADD CONSTRAINT fk_sessions_user 
FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
若已有历史数据（如之前使用了openid作为user_id），需先迁移数据：将openid作为username插入users表（密码可临时设为随机哈希，或要求用户重置）。但新系统可从头开始，按新设计创建用户。

## 6. 前端页面与交互简述
6.1 对话页
底部输入框+发送按钮。

未登录时点击发送 → 弹出登录提示框。

已登录时正常发送。

6.2 登录/注册页（或弹窗）
包含两个Tab：登录、注册。

登录Tab：用户名输入框、密码输入框、登录按钮。

注册Tab：用户名输入框、密码输入框、确认密码输入框、昵称输入框（可选）、注册按钮。

注册成功后自动登录（返回token），或注册后跳转登录页手动登录（推荐注册后自动登录，简化流程）。

6.3 自动重发消息
在触发登录弹窗前，将用户输入的消息暂存。

登录/注册成功后，自动使用该消息调用发送接口。

## 7. 安全注意事项
密码传输必须使用HTTPS，防止中间人窃听。

后端密码存储使用bcrypt加盐哈希，禁止明文存储。

JWT密钥定期更换，并设置合理的过期时间。

前端token存储在uni.setStorageSync（小程序）或localStorage（Web），避免XSS风险（但需注意localStorage的XSS问题，可考虑httpOnly cookie，但小程序不支持，故使用存储+请求头方式）。

## 8. 测试与模拟
开发阶段可提供测试账号：用户名 test，密码 123456。

注册时需校验用户名唯一性。


