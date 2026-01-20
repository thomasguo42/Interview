from typing import List, Dict


class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        pairs_by_b = [[] for _ in range(n + 1)]
        a_to_bs: Dict[int, List[int]] = {}
        for a, b in conflictingPairs:
            if a > b:
                a, b = b, a
            pairs_by_b[b].append(a)
            a_to_bs.setdefault(a, []).append(b)

        highest = [0] * (n + 1)
        second = [0] * (n + 1)
        count_highest = [0] * (n + 1)

        max_a = 0
        second_a = 0
        count = 0
        for r in range(1, n + 1):
            for a in pairs_by_b[r]:
                if a > max_a:
                    second_a = max_a
                    max_a = a
                    count = 1
                elif a == max_a:
                    count += 1
                elif a > second_a:
                    second_a = a
            highest[r] = max_a
            second[r] = second_a
            count_highest[r] = count

        base_total = 0
        diff = [0] * (n + 1)
        for r in range(1, n + 1):
            base_total += r - highest[r]
            if count_highest[r] == 1:
                diff[r] = highest[r] - second[r]

        prefix = [0] * (n + 1)
        for r in range(1, n + 1):
            prefix[r] = prefix[r - 1] + diff[r]

        max_segment_end: Dict[int, int] = {}
        current = highest[1] if n >= 1 else 0
        start = 1
        for r in range(2, n + 1):
            if highest[r] != current:
                max_segment_end[current] = r - 1
                current = highest[r]
                start = r
        if n >= 1:
            max_segment_end[current] = n

        best_delta = 0
        for a, bs in a_to_bs.items():
            bs.sort()
            first_b = bs[0]
            if bs.count(first_b) != 1:
                continue
            if highest[first_b] != a or count_highest[first_b] != 1:
                continue
            second_b = bs[1] if len(bs) > 1 else n + 1
            r_end = max_segment_end.get(a, 0)
            range_end = min(r_end, second_b - 1)
            if range_end >= first_b:
                delta = prefix[range_end] - prefix[first_b - 1]
                if delta > best_delta:
                    best_delta = delta

        return base_total + best_delta
