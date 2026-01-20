from typing import List


class Solution:
    def ipToCIDR(self, ip: str, n: int) -> List[str]:
        def ip_to_int(addr: str) -> int:
            parts = list(map(int, addr.split('.')))
            val = 0
            for p in parts:
                val = (val << 8) + p
            return val

        def int_to_ip(val: int) -> str:
            return '.'.join(str((val >> (8 * i)) & 255) for i in reversed(range(4)))

        start = ip_to_int(ip)
        res = []
        while n > 0:
            lowbit = start & -start
            if lowbit == 0:
                lowbit = 1 << 32
            size = lowbit
            while size > n:
                size >>= 1
            mask = 32 - (size.bit_length() - 1)
            res.append(f"{int_to_ip(start)}/{mask}")
            start += size
            n -= size
        return res
