# ✅ **BƯỚC 1 — Cài Python + ngrok trên Windows**

Dùng winget:

```bash
winget install --id Python.Python.3 --source winget
winget install --id Ngrok.Ngrok --source winget
```

---

# ✅ **BƯỚC 2 — Lấy authtoken của ngrok**

Vào: [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)

Chạy lệnh:

```bash
ngrok config add-authtoken <YOUR_TOKEN>
```

---

# ✅ **BƯỚC 3 — Tạo server webhook Windows (tự shutdown)**

Tạo folder bất kỳ, ví dụ `C:\ColabShutdown`.

### 📌 1. Tạo file `shutdown_listener.py`

Copy nguyên cái này:

```python
from flask import Flask, request
import os

app = Flask(__name__)

@app.post("/done")
def done():
    print(">>> Colab báo: Training xong rồi! Chuẩn bị tắt máy...")
    os.system("shutdown /s /t 60")  # tắt sau 60 giây, đổi về 0 nếu muốn tắt ngay
    return {"status": "received"}, 200

app.run(host="0.0.0.0", port=5005)
```

### 📌 2. Cài Flask

```bash
pip install flask
```

### 📌 3. Chạy listener

```bash
python shutdown_listener.py
```

---

# ✅ **BƯỚC 4 — Mở cổng webhook bằng ngrok**

Trong 1 cửa sổ CMD khác chạy:

```bash
ngrok http 5005
```

Bạn sẽ thấy một URL dạng:

```
https://abcd1234.ngrok.io
```

Webhook endpoint chính là:

```
https://abcd1234.ngrok.io/done
```

---

# ✅ **BƯỚC 5 — Thêm callback vào cuối Colab Notebook**

Khi train xong → gọi về Windows → Windows tắt máy 💀💤

Ngay dưới cell cuối cùng bạn thêm:

```python
import requests

WEBHOOK_URL = "https://abcd1234.ngrok.io/done"  # đổi URL của bạn

try:
    r = requests.post(WEBHOOK_URL)
    print("Đã gửi tín hiệu shutdown về PC!")
except Exception as e:
    print("Không gửi được webhook:", e)
```