from typing import List


class Solution:
    def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
        m = len(board)
        n = len(board[0]) if board else 0
        changed = True
        while changed:
            changed = False
            crush = [[False] * n for _ in range(m)]
            for i in range(m):
                for j in range(n - 2):
                    v = abs(board[i][j])
                    if v != 0 and v == abs(board[i][j + 1]) == abs(board[i][j + 2]):
                        crush[i][j] = crush[i][j + 1] = crush[i][j + 2] = True
            for j in range(n):
                for i in range(m - 2):
                    v = abs(board[i][j])
                    if v != 0 and v == abs(board[i + 1][j]) == abs(board[i + 2][j]):
                        crush[i][j] = crush[i + 1][j] = crush[i + 2][j] = True
            for i in range(m):
                for j in range(n):
                    if crush[i][j]:
                        board[i][j] = 0
                        changed = True
            for j in range(n):
                write = m - 1
                for i in range(m - 1, -1, -1):
                    if board[i][j] != 0:
                        board[write][j] = board[i][j]
                        write -= 1
                for i in range(write, -1, -1):
                    board[i][j] = 0
        return board
