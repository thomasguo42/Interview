from typing import List

def numIslands(grid: List[List[str]]) -> int:
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    seen = [[False] * n for _ in range(m)]

    def dfs(sr, sc):
        stack = [(sr, sc)]
        seen[sr][sc] = True
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and not seen[nr][nc] and grid[nr][nc] == "1":
                    seen[nr][nc] = True
                    stack.append((nr, nc))

    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1" and not seen[i][j]:
                count += 1
                dfs(i, j)
    return count
