#!/usr/bin/env python3
import sys

current_ip = None
count = 0

for line in sys.stdin:
    ip, num = line.strip().split('\t')
    num = int(num)
    if ip != current_ip:
        if current_ip:
            print(f"{current_ip}\t{count}")
        current_ip = ip
        count = num
    else:
        count += num

if current_ip:
    print(f"{current_ip}\t{count}")
