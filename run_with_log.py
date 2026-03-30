#!/usr/bin/env python3
"""
Rulează main.py și salvează output-ul în fișier și consolă.
"""

import sys
import subprocess
import threading
import time

def read_output(pipe, log_file, print_output=True):
    """Citește output-ul și îl scrie în fișier și consolă."""
    for line in iter(pipe.readline, ''):
        if print_output:
            print(line, end='')
        log_file.write(line)
        log_file.flush()
    pipe.close()

def main():
    log_filename = f"server_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    print(f"Pornire server... Log salvat în: {log_filename}")
    print("=" * 60)
    
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        # Pornește procesul
        process = subprocess.Popen(
            [sys.executable, 'main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Thread pentru citire output
        thread = threading.Thread(
            target=read_output,
            args=(process.stdout, log_file, True)
        )
        thread.daemon = True
        thread.start()
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nOprire server...")
            process.terminate()
            process.wait()

if __name__ == "__main__":
    main()