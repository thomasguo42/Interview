import java.util.*;

public class Solution {
    public static int numIslands(String[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        int rows = grid.length;
        int cols = grid[0].length;
        boolean[][] seen = new boolean[rows][cols];
        int count = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (!seen[r][c] && "1".equals(grid[r][c])) {
                    count++;
                    dfs(grid, seen, r, c);
                }
            }
        }
        return count;
    }

    private static void dfs(String[][] grid, boolean[][] seen, int r, int c) {
        int rows = grid.length;
        int cols = grid[0].length;
        if (r < 0 || c < 0 || r >= rows || c >= cols) return;
        if (seen[r][c] || !"1".equals(grid[r][c])) return;
        seen[r][c] = true;
        dfs(grid, seen, r + 1, c);
        dfs(grid, seen, r - 1, c);
        dfs(grid, seen, r, c + 1);
        dfs(grid, seen, r, c - 1);
    }
}
