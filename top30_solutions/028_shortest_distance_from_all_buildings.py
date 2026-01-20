from typing import List
from collections import deque


class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        if not grid:
            return -1
        m, n = len(grid), len(grid[0])
        dist = [[0] * n for _ in range(m)]
        reach = [[0] * n for _ in range(m)]
        buildings = [(i, j) for i in range(m) for j in range(n) if grid[i][j] == 1]
        for bx, by in buildings:
            q = deque([(bx, by, 0)])
            seen = [[False] * n for _ in range(m)]
            seen[bx][by] = True
            while q:
                x, y, d = q.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and not seen[nx][ny] and grid[nx][ny] == 0:
                        seen[nx][ny] = True
                        dist[nx][ny] += d + 1
                        reach[nx][ny] += 1
                        q.append((nx, ny, d + 1))
        best = min(
            (dist[i][j] for i in range(m) for j in range(n) if grid[i][j] == 0 and reach[i][j] == len(buildings)),
            default=float('inf')
        )
        return -1 if best == float('inf') else best
