from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        half = (m + n + 1) // 2
        while left <= right:
            i = (left + right) // 2
            j = half - i
            left_max = nums1[i - 1] if i > 0 else float('-inf')
            right_min = nums1[i] if i < m else float('inf')
            left_max2 = nums2[j - 1] if j > 0 else float('-inf')
            right_min2 = nums2[j] if j < n else float('inf')
            if left_max <= right_min2 and left_max2 <= right_min:
                if (m + n) % 2:
                    return float(max(left_max, left_max2))
                return (max(left_max, left_max2) + min(right_min, right_min2)) / 2.0
            if left_max > right_min2:
                right = i - 1
            else:
                left = i + 1
        return 0.0
