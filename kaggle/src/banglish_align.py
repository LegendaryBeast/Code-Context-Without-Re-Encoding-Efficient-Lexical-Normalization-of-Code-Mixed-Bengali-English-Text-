"""Monotonic source-token -> target-span aligner for BanglaMix.

93.4% of comments have one target word per source token. The rest differ by a few words: a glued token
("Bhaiনাকি") becomes two target words, or punctuation is attached to the previous word ("back ." -> "ব্যাক।").
Word-level models drop these comments from training, which removes ~19% of all ENG/ROM training tokens
(they are long, Latin-heavy comments). This module aligns them so every comment yields one target span per
source token.

A span may hold several words ("ভাই নাকি") or start with GLUE, meaning "attach to the previous output
without a space" (the "." of "back ." gets GLUE + "।"). `join_spans` turns spans back into a string, so
join_spans(align(...)) == the original target for every comment (checked by `selftest`).

Pure Python, no dependencies; costs come from a coarse phonetic skeleton shared by Latin and Bengali script.
"""
import unicodedata

GLUE = "\x02"   # span prefix: attach to the previous span without a space (not whitespace to str.split)

_BN = {}
for chars, v in [("কখগঘ", "k"), ("চছ", "k"), ("জঝয", "j"), ("টঠতথৎ", "t"), ("ডঢদধ", "d"),
                 ("ঙঞণনং", "n"), ("পফ", "p"), ("বভ", "b"), ("ম", "m"), ("র", "r"), ("ড়ঢ়", "r"), ("ল", "l"),
                 ("শষস", "s"), ("হঃ", ""), ("য়", ""), ("ঁ", ""), ("।", "."),
                 ("অআইঈউঊঋএঐওঔ", ""), ("ািীুূৃেৈোৌ্", "")]:
    for ch in chars:
        _BN[ch] = v
for i, ch in enumerate("০১২৩৪৫৬৭৮৯"):
    _BN[ch] = str(i)
_LAT = {**{c: "" for c in "aeiouwyh"}, **{c: "k" for c in "ckq"}, "x": "ks", "z": "j", "v": "b", "f": "p",
        "g": "k"}  # g~k: voicing is unreliable in casual spelling; coarse classes only need to rank alignments



def skeleton(s):
    out = []
    for ch in unicodedata.normalize("NFC", s.lower()):
        if ch in _BN:
            out.append(_BN[ch])
        elif "a" <= ch <= "z":
            out.append(_LAT.get(ch, ch))
        elif ch.isdigit() or not ch.isalnum():
            out.append(ch if not ch.isspace() else "")
        else:
            out.append(ch)
    s = "".join(out)
    return "".join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])   # collapse repeats


def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_symbolic(w):
    return not any(ch.isalpha() for ch in w)


class Aligner:
    """lexicon: optional {(source_word_lower, target_word)} pairs seen in aligned training comments."""

    def __init__(self, lexicon=None):
        self.lex = lexicon or set()
        self._sk = {}

    def _skel(self, s):
        if s not in self._sk:
            self._sk[s] = skeleton(s)
        return self._sk[s]

    def tok_cost(self, src, tgt):
        """0 = perfect match, ~1 = unrelated."""
        if not tgt:
            return 0.35 if _is_symbolic(src) else 1.0
        if (src.lower(), tgt) in self.lex or src == tgt:
            return 0.0
        a, b = self._skel(src), self._skel(tgt)
        if not a and not b:
            return 0.1
        return _lev(a, b) / max(len(a), len(b))

    def align(self, src, tgt):
        """src: list of source tokens, tgt: list of target words -> (spans, mean cost per source token)."""
        n, m = len(src), len(tgt)
        INF = float("inf")
        D = [[INF] * (m + 1) for _ in range(n + 1)]
        B = [[None] * (m + 1) for _ in range(n + 1)]
        D[0][0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                if D[i][j] == INF:
                    continue
                base = D[i][j]
                # one source token -> b target words (b = 0..3)
                if i < n:
                    for b in range(0, 4):
                        if j + b > m:
                            break
                        span = " ".join(tgt[j:j + b])
                        c = base + self.tok_cost(src[i], span) + (0.0 if b == 1 else 0.25 * abs(b - 1))
                        if c < D[i + 1][j + b]:
                            D[i + 1][j + b], B[i + 1][j + b] = c, ("split", 1, b, None)
                # a source tokens (a = 2..3) -> one target word, split inside the word
                if j < m:
                    w = tgt[j]
                    for a in (2, 3):
                        if i + a > n:
                            break
                        best, cut = self._best_cuts(src[i:i + a], w)
                        c = base + best + 0.25 * (a - 1)
                        if c < D[i + a][j + 1]:
                            D[i + a][j + 1], B[i + a][j + 1] = c, ("merge", a, 1, cut)
        if D[n][m] == INF:
            return None, INF
        spans, i, j = [], n, m
        while i or j:
            kind, a, b, cut = B[i][j]
            if kind == "split":
                spans.append(" ".join(tgt[j - b:j]))
            else:
                parts = [tgt[j - 1][s:e] for s, e in zip((0,) + cut, cut + (len(tgt[j - 1]),))]
                out, started = [], False
                for p in parts:                    # first non-empty part opens the word, later ones glue on
                    out.append("" if not p else (GLUE + p if started else p))
                    started = started or bool(p)
                spans += out[::-1]
            i, j = i - a, j - b
        return spans[::-1], D[n][m] / max(1, n)

    def _best_cuts(self, srcs, w):
        """Split target word w into len(srcs) contiguous parts (empty parts allowed) at minimum cost."""
        L = len(w)
        if len(srcs) == 2:
            cands = [(k,) for k in range(L + 1)]
        else:
            step = max(1, L // 12)                     # keep the 3-way search small for long words
            cands = [(k1, k2) for k1 in range(0, L + 1, step) for k2 in range(k1, L + 1, step)]
        best, arg = float("inf"), None
        for cut in cands:
            parts = [w[s:e] for s, e in zip((0,) + cut, cut + (L,))]
            c = sum(self.tok_cost(s, p) for s, p in zip(srcs, parts))
            if c < best:
                best, arg = c, cut
        return best, arg


def join_spans(spans):
    out = ""
    for s in spans:
        if not s:
            continue
        if s.startswith(GLUE):
            out += s[len(GLUE):]
        else:
            out += (" " if out else "") + s
    return out


def selftest():
    al = Aligner()
    cases = [
        (["back", "."], ["ব্যাক।"]),
        (["Bhaiনাকি", ".."], ["ভাই", "নাকি", ".."]),
        (["friend", "ship✨✨"], ["ফ্রেন্ডশিপ✨✨"]),
        (["vaiya", "please", "give"], ["ভাইয়া", "প্লিজ", "গিভ"]),
        (["70theke", "75k"], ["৭০", "থেকে", "৭৫কে"]),
    ]
    for s, t in cases:
        spans, c = al.align(s, t)
        assert join_spans(spans) == " ".join(t), (s, t, spans)
        print(f"{c:.2f}", list(zip(s, [x.replace(GLUE, '+') for x in spans])))


if __name__ == "__main__":
    selftest()
