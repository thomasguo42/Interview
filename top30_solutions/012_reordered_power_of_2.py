class Solution:
    def reorderedPowerOf2(self, n: int) -> bool:
        def count_digits(x: int) -> tuple:
            counts = [0] * 10
            for ch in str(x):
                counts[ord(ch) - 48] += 1
            return tuple(counts)

        target = count_digits(n)
        for i in range(31):
            if count_digits(1 << i) == target:
                return True
        return False
