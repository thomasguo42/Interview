from typing import List

def numIslands(grid: List[List[str]]) -> int:
    # Wrong: counts every land cell instead of components
    return sum(row.count("1") for row in grid)
