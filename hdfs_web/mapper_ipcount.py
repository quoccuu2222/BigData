#!/usr/bin/env python3
import sys

for line in sys.stdin:
    parts = line.strip().split()
    if len(parts) > 0:
        ip = parts[0]
        print(f"{ip}\t1")
