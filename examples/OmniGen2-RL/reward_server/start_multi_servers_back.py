#!/usr/bin/env python3
"""
Script to launch multi-GPU servers.
Start specified server instances on all GPUs.
"""

import subprocess
import time
import os
import signal
import sys
import argparse
from multiprocessing import Process

# Server configuration
NUM_GPUS = 8
BASE_PORT = 18086
# HOST = "127.0.0.1"
HOST = "0.0.0.0"
MODEL_PATH = "/share/project/jiahao/geneval/models"

# Store subprocesses
server_processes = []

def start_server(args, worker_idx, num_gpus_per_worker, port, server_script, unknown_args):
    """Start server on the specified GPU(s)"""
    cmd = [
        sys.executable, server_script,
        "--port", str(port),
        "--host", HOST,
        *unknown_args,
    ]
    
    # Add model-path for GenEval server
    if "geneval" in server_script:
        cmd.extend(["--model-path", MODEL_PATH])
    # Add use-gpu for OCR server
    elif "ocr" in server_script:
        cmd.extend(["--use-gpu"])
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(worker_idx * num_gpus_per_worker, (worker_idx + 1) * num_gpus_per_worker))
    
    # Create log directory
    os.makedirs("./logs", exist_ok=True)
    log_file = f"./logs/{args.log_name}_worker_{worker_idx}.log"
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
        print(f"📝 Worker {worker_idx} log file: {log_file}")
        return process
    except Exception as e:
        print(f"❌ Failed to start server for Worker {worker_idx}: {e}")
        return None

def signal_handler(signum, frame):
    """Handle exit signals"""
    print("\n🛑 Received exit signal, shutting down all servers...")
    for i, process in enumerate(server_processes):
        if process and process.poll() is None:
            print(f"Shutting down server Worker {i}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    sys.exit(0)

def check_server_logs(args, worker_idx):
    """Check server log file"""
    log_file = f"./logs/{args.log_name}_worker_{worker_idx}.log"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                if content:
                    print(f"\n📄 Worker {worker_idx} error log:")
                    print(content[-1000:])  # Show last 1000 characters
        except Exception as e:
            print(f"Failed to read log file {log_file}: {e}")

def main():
    global NUM_GPUS, BASE_PORT
    
    # Argument parsing
    parser = argparse.ArgumentParser(description='Launch multi-GPU servers')
    parser.add_argument('server_script', help='Server script filename to launch')
    parser.add_argument('--base_port', type=int, default=BASE_PORT, help=f'Base port (default: {BASE_PORT})')
    parser.add_argument('--log_name', type=str, default=None, help='Log name prefix (default: server)')
    parser.add_argument('--num_gpus_per_worker', type=int, default=1, help=f'Number of GPUs per worker (default: 1)')
    
    args, unknown_args = parser.parse_known_args()
    
    # Add .py suffix if missing
    server_script = args.server_script
    if not server_script.endswith('.py'):
        server_script += '.py'
    
    BASE_PORT = args.base_port
    
    # Check if script exists
    if not os.path.exists(server_script):
        print(f"❌ Script file {server_script} does not exist")
        sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Detect available GPUs
    import torch
    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
    else:
        num_available_gpus = 0
    
    num_workers = num_available_gpus // args.num_gpus_per_worker
    
    print(f"🔥 Launching {num_workers} {server_script} server(s)")
    print(f"📍 Base port: {BASE_PORT}")
    print(f"🖥️  Host: {HOST}")
    if "geneval" in server_script:
        print(f"📁 Model path: {MODEL_PATH}")
    print("-" * 50)
    
    if num_workers == 0:
        print("❌ No available GPUs, exiting")
        sys.exit(1)
    
    # Start all servers
    for worker_idx in range(num_workers):
        port = BASE_PORT + worker_idx
        process = start_server(args, worker_idx, args.num_gpus_per_worker, port, server_script, unknown_args)
        server_processes.append(process)
        time.sleep(3)  # Add delay to avoid resource contention
    
    print(f"\n✅ Attempted to launch {num_workers} server(s)")
    print("Server list:")
    for worker_idx in range(num_workers):
        port = BASE_PORT + worker_idx
        print(f"  GPU {worker_idx}: http://{HOST}:{port}")
    
    print(f"\n⏳ Waiting for servers to start...")
    time.sleep(10)  # Wait for servers to start
    
    # Check server status
    running_servers = 0
    for i, process in enumerate(server_processes):
        if process and process.poll() is None:
            running_servers += 1
            print(f"✅ Worker {i} server is running")
        else:
            print(f"❌ Worker {i} server failed to start")
            check_server_logs(args, i)
    
    if running_servers == 0:
        print("❌ No servers started successfully")
        sys.exit(1)
    
    print(f"\n🎉 Successfully started {running_servers} server(s)")
    print("⏳ Servers are running... (Press Ctrl+C to exit)")
    
    # Monitor server status
    try:
        while True:
            time.sleep(10)  # Check interval
            # Check for crashed servers
            for i, process in enumerate(server_processes):
                if process and process.poll() is not None:
                    print(f"⚠️  Server Worker {i} exited (return code: {process.returncode})")
                    check_server_logs(args, i)
                    # Set exited process to None to avoid duplicate reporting
                    server_processes[i] = None
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main() 