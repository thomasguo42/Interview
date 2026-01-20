class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ''
        start = end = 0
        for i in range(len(s)):
            for a, b in ((i, i), (i, i + 1)):
                left, right = a, b
                while left >= 0 and right < len(s) and s[left] == s[right]:
                    if right - left > end - start:
                        start, end = left, right
                    left -= 1
                    right += 1
        return s[start:end + 1]
