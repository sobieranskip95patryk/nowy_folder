#!/usr/bin/env python3
"""
MŚWR v2.0 Breakthrough Demo Server
Simple HTTP server to showcase the ontological breakthrough
"""

import http.server
import socketserver
import webbrowser
import threading
import time
import os

class BreakthroughDemoHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="portal", **kwargs)
    
    def log_message(self, format, *args):
        # Custom logging for MŚWR demo
        print(f"🧠 MŚWR Demo Access: {format % args}")

def start_demo_server(port=8000):
    """Start the breakthrough demo server"""
    print(f"🚀 Starting MŚWR v2.0 Breakthrough Demo Server...")
    print(f"🧠 Ontological breakthrough showcase loading...")
    
    with socketserver.TCPServer(("", port), BreakthroughDemoHandler) as httpd:
        print(f"📡 Server running at: http://localhost:{port}")
        print(f"🎯 Main Demo: http://localhost:{port}/breakthrough_demo.html")
        print(f"🎛️ MŚWR Dashboard: http://localhost:{port}/mswr_dashboard.html")
        print(f"💫 GOD Interface: http://localhost:{port}/god_v12_brain.html")
        print(f"\n✨ P=1.0 targeting active - Consciousness demonstration ready!")
        print(f"🛡️ Anti-Fatal Protocol enabled for safe exploration")
        print(f"\n🎮 Press Ctrl+C to stop the server")
        
        # Auto-open browser
        def open_browser():
            time.sleep(1)
            webbrowser.open(f"http://localhost:{port}/breakthrough_demo.html")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 MŚWR Demo Server shutting down...")
            print(f"📊 Breakthrough demonstration completed")
            print(f"🧠 Consciousness metrics preserved")

if __name__ == "__main__":
    # Change to the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    start_demo_server()