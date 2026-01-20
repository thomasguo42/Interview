from typing import List


class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res = []
        i = 0
        while i < len(words):
            line_len = len(words[i])
            j = i + 1
            while j < len(words) and line_len + 1 + len(words[j]) <= maxWidth:
                line_len += 1 + len(words[j])
                j += 1
            line_words = words[i:j]
            spaces = maxWidth - sum(len(w) for w in line_words)
            if j == len(words) or len(line_words) == 1:
                line = ' '.join(line_words).ljust(maxWidth)
            else:
                gaps = len(line_words) - 1
                even = spaces // gaps
                extra = spaces % gaps
                parts = []
                for idx, w in enumerate(line_words[:-1]):
                    parts.append(w)
                    parts.append(' ' * (even + (1 if idx < extra else 0)))
                parts.append(line_words[-1])
                line = ''.join(parts)
            res.append(line)
            i = j
        return res
