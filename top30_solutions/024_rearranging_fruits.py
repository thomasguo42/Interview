from typing import List
from collections import Counter


class Solution:
    def minCost(self, basket1: List[int], basket2: List[int]) -> int:
        c1 = Counter(basket1)
        c2 = Counter(basket2)
        all_keys = set(c1) | set(c2)
        surplus = []
        deficit = []
        for k in all_keys:
            total = c1.get(k, 0) + c2.get(k, 0)
            if total % 2:
                return -1
            diff = c1.get(k, 0) - c2.get(k, 0)
            if diff > 0:
                surplus.extend([k] * (diff // 2))
            elif diff < 0:
                deficit.extend([k] * ((-diff) // 2))
        if not surplus:
            return 0
        surplus.sort()
        deficit.sort(reverse=True)
        global_min = min(all_keys)
        cost = 0
        for a, b in zip(surplus, deficit):
            cost += min(a, b, 2 * global_min)
        return cost
