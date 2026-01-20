from typing import List


class Solution:
    def judgePoint24(self, nums: List[int]) -> bool:
        eps = 1e-6

        def solve(arr: List[float]) -> bool:
            if len(arr) == 1:
                return abs(arr[0] - 24.0) < eps
            for i in range(len(arr)):
                for j in range(len(arr)):
                    if i == j:
                        continue
                    next_arr = [arr[k] for k in range(len(arr)) if k not in (i, j)]
                    a, b = arr[i], arr[j]
                    for v in (a + b, a - b, b - a, a * b):
                        next_arr.append(v)
                        if solve(next_arr):
                            return True
                        next_arr.pop()
                    if abs(b) > eps:
                        next_arr.append(a / b)
                        if solve(next_arr):
                            return True
                        next_arr.pop()
                    if abs(a) > eps:
                        next_arr.append(b / a)
                        if solve(next_arr):
                            return True
                        next_arr.pop()
            return False

        return solve([float(x) for x in nums])
