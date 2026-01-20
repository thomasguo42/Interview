from typing import List


# The Sea API is provided by the platform.
# class Sea:
#     def hasShips(self, topRight: List[int], bottomLeft: List[int]) -> bool:


class Solution:
    def countShips(self, sea: 'Sea', topRight: List[int], bottomLeft: List[int]) -> int:
        if topRight[0] < bottomLeft[0] or topRight[1] < bottomLeft[1]:
            return 0
        if not sea.hasShips(topRight, bottomLeft):
            return 0
        if topRight == bottomLeft:
            return 1
        mid_x = (topRight[0] + bottomLeft[0]) // 2
        mid_y = (topRight[1] + bottomLeft[1]) // 2
        return (
            self.countShips(sea, [mid_x, mid_y], bottomLeft) +
            self.countShips(sea, [mid_x, topRight[1]], [bottomLeft[0], mid_y + 1]) +
            self.countShips(sea, [topRight[0], mid_y], [mid_x + 1, bottomLeft[1]]) +
            self.countShips(sea, topRight, [mid_x + 1, mid_y + 1])
        )
