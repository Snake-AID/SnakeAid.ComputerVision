from flask import Flask, request
import os
import subprocess
import time
import threading
import requests
from rich.console import Console
from rich.panel import Panel

NGROK_PORT = 5005
console = Console()
app = Flask(__name__)

@app.post("/done")
def done():
    console.print("[bold red]>>> Colab báo: Training xong, chuẩn bị tắt máy...[/]")
    os.system("shutdown /s /t 60")  # đổi 0 nếu muốn tắt ngay
    return {"status": "received"}, 200

def start_ngrok(port: int):
    # chạy ngrok http <port> background
    subprocess.Popen(
        ["ngrok", "http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    public_url = None
    for _ in range(15):
        try:
            resp = requests.get("http://127.0.0.1:4040/api/tunnels").json()
            tunnels = resp.get("tunnels", [])
            if tunnels:
                public_url = tunnels[0]["public_url"]
                break
        except Exception:
            pass
        time.sleep(1)

    if public_url:
        console.print(
            Panel.fit(
                f"[bold cyan]{public_url}/done[/]",
                title="🌐 NGROK WEBHOOK URL",
                border_style="green",
            )
        )
    else:
        console.print("[bold red]Không lấy được URL từ ngrok, kiểm tra lại.[/]")

if __name__ == "__main__":
    t = threading.Thread(target=start_ngrok, args=(NGROK_PORT,), daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=NGROK_PORT)