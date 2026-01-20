from typing import List
from collections import defaultdict


class Solution:
    def numberWays(self, hats: List[List[int]]) -> int:
        mod = 10 ** 9 + 7
        n = len(hats)
        hat_to_people = defaultdict(list)
        for i, hs in enumerate(hats):
            for h in hs:
                hat_to_people[h].append(i)
        max_hat = max(hat_to_people) if hat_to_people else 0
        dp = [0] * (1 << n)
        dp[0] = 1
        for h in range(1, max_hat + 1):
            if h not in hat_to_people:
                continue
            next_dp = dp[:]
            for mask in range(1 << n):
                if dp[mask] == 0:
                    continue
                for p in hat_to_people[h]:
                    if mask & (1 << p) == 0:
                        nm = mask | (1 << p)
                        next_dp[nm] = (next_dp[nm] + dp[mask]) % mod
            dp = next_dp
        return dp[(1 << n) - 1]
