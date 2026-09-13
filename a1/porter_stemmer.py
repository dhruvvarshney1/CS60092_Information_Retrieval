"""
porter_stemmer.py
-----------------
A from-scratch implementation of the Porter stemming algorithm
(M.F. Porter, "An algorithm for suffix stripping", Program, 14(3),
pp. 130-137, 1980).

No external / IR library is used -- only plain Python string handling.

Public interface
----------------
    stem(word) -> str          stem a single lower-case word
    PorterStemmer().stem(w)    object interface (identical result)

The implementation follows the five phases of the original algorithm:

    Step 1a/1b/1c : plurals and past participles          (sses -> ss, ies -> i,
                                                           eed -> ee, -ed, -ing, y -> i)
    Step 2        : double suffixes -> single             (ational -> ate, ...)
    Step 3        : -ic-, -full, -ness ...                (icate -> ic, ful -> "")
    Step 4        : -ant, -ence ... removal               (m > 1)
    Step 5a/5b    : tidy-up (final -e, double -ll)
"""

VOWELS = frozenset("aeiou")


class PorterStemmer(object):
    """Porter stemmer.  `b[k0:k+1]` is the word currently being stemmed."""

    def __init__(self):
        self.b = ""   # buffer holding the word
        self.k = 0    # index of the last character of the word
        self.k0 = 0   # index of the first character (always 0 here)
        self.j = 0    # general offset used by the step functions

    # ------------------------------------------------------------------ #
    # Primitive tests                                                     #
    # ------------------------------------------------------------------ #
    def _cons(self, i):
        """True if b[i] is a consonant."""
        ch = self.b[i]
        if ch in VOWELS:
            return False
        if ch == 'y':
            # 'y' is a consonant iff the preceding letter is a vowel
            return True if i == self.k0 else (not self._cons(i - 1))
        return True

    def _m(self):
        """Measure m of b[k0:j+1]: the number of VC sequences."""
        n = 0
        i = self.k0
        # skip the initial (optional) consonant sequence
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            # a vowel sequence followed by a consonant sequence => n += 1
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self):
        """True if b[k0:j+1] contains a vowel."""
        for i in range(self.k0, self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, j):
        """True if b[j-1] == b[j] and both are consonants."""
        if j < self.k0 + 1:
            return False
        if self.b[j] != self.b[j - 1]:
            return False
        return self._cons(j)

    def _cvc(self, i):
        """True if b[i-2:i+1] is consonant-vowel-consonant and the final
        consonant is not w, x or y (used to restore a final 'e')."""
        if i < self.k0 + 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    # ------------------------------------------------------------------ #
    # Suffix helpers                                                      #
    # ------------------------------------------------------------------ #
    def _ends(self, s):
        """True if the word ends with s; sets j to just before the suffix."""
        length = len(s)
        if length > (self.k - self.k0 + 1):
            return False
        if self.b[self.k - length + 1:self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s):
        """Replace b[j+1:k+1] by s."""
        self.b = self.b[:self.j + 1] + s
        self.k = self.j + len(s)

    def _r(self, s):
        """_setto(s) but only when m() > 0."""
        if self._m() > 0:
            self._setto(s)

    # ------------------------------------------------------------------ #
    # The five steps                                                      #
    # ------------------------------------------------------------------ #
    def _step1ab(self):
        # Step 1a -- plurals
        if self.b[self.k] == 's':
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != 's':
                self.k -= 1
        # Step 1b -- past participles / progressive
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                if self.b[self.k] not in "lsz":
                    self.k -= 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self):
        """Terminal y -> i when the stem contains a vowel."""
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[:self.k] + 'i'

    def _step2(self):
        """Map double suffixes to single ones (m > 0)."""
        if self.k < self.k0 + 1:
            return
        ch = self.b[self.k - 1]
        if ch == 'a':
            if self._ends("ational"):   self._r("ate")
            elif self._ends("tional"):  self._r("tion")
        elif ch == 'c':
            if self._ends("enci"):      self._r("ence")
            elif self._ends("anci"):    self._r("ance")
        elif ch == 'e':
            if self._ends("izer"):      self._r("ize")
        elif ch == 'l':
            if self._ends("bli"):       self._r("ble")
            elif self._ends("alli"):    self._r("al")
            elif self._ends("entli"):   self._r("ent")
            elif self._ends("eli"):     self._r("e")
            elif self._ends("ousli"):   self._r("ous")
        elif ch == 'o':
            if self._ends("ization"):   self._r("ize")
            elif self._ends("ation"):   self._r("ate")
            elif self._ends("ator"):    self._r("ate")
        elif ch == 's':
            if self._ends("alism"):     self._r("al")
            elif self._ends("iveness"): self._r("ive")
            elif self._ends("fulness"): self._r("ful")
            elif self._ends("ousness"): self._r("ous")
        elif ch == 't':
            if self._ends("aliti"):     self._r("al")
            elif self._ends("iviti"):   self._r("ive")
            elif self._ends("biliti"):  self._r("ble")
        elif ch == 'g':
            if self._ends("logi"):      self._r("log")

    def _step3(self):
        """-ic-, -full, -ness etc. (m > 0)."""
        ch = self.b[self.k]
        if ch == 'e':
            if self._ends("icate"):     self._r("ic")
            elif self._ends("ative"):   self._r("")
            elif self._ends("alize"):   self._r("al")
        elif ch == 'i':
            if self._ends("iciti"):     self._r("ic")
        elif ch == 'l':
            if self._ends("ical"):      self._r("ic")
            elif self._ends("ful"):     self._r("")
        elif ch == 's':
            if self._ends("ness"):      self._r("")

    def _step4(self):
        """Remove -ant, -ence, ... when m > 1."""
        if self.k < self.k0 + 1:
            return
        ch = self.b[self.k - 1]
        if ch == 'a':
            if not self._ends("al"):        return
        elif ch == 'c':
            if not (self._ends("ance") or self._ends("ence")):   return
        elif ch == 'e':
            if not self._ends("er"):        return
        elif ch == 'i':
            if not self._ends("ic"):        return
        elif ch == 'l':
            if not (self._ends("able") or self._ends("ible")):   return
        elif ch == 'n':
            if not (self._ends("ant") or self._ends("ement")
                    or self._ends("ment") or self._ends("ent")): return
        elif ch == 'o':
            if self._ends("ion") and self.j >= self.k0 and self.b[self.j] in "st":
                pass
            elif not self._ends("ou"):      return
        elif ch == 's':
            if not self._ends("ism"):       return
        elif ch == 't':
            if not (self._ends("ate") or self._ends("iti")):     return
        elif ch == 'u':
            if not self._ends("ous"):       return
        elif ch == 'v':
            if not self._ends("ive"):       return
        elif ch == 'z':
            if not self._ends("ize"):       return
        else:
            return
        if self._m() > 1:
            self.k = self.j

    def _step5(self):
        """Remove a final -e (5a) and reduce a double -ll (5b)."""
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == 'l' and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    # ------------------------------------------------------------------ #
    # Driver                                                              #
    # ------------------------------------------------------------------ #
    def stem(self, word):
        """Return the Porter stem of `word` (assumed already lower-cased)."""
        if word is None:
            return word
        # words of one or two letters are returned unchanged
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[self.k0:self.k + 1]


# module-level convenience wrapper -------------------------------------- #
_STEMMER = PorterStemmer()
_CACHE = {}


def stem(word):
    """Memoised single-word stemming (the corpus repeats tokens heavily)."""
    s = _CACHE.get(word)
    if s is None:
        s = _STEMMER.stem(word)
        _CACHE[word] = s
    return s


if __name__ == "__main__":
    tests = ["caresses", "ponies", "ties", "caress", "cats", "feed", "agreed",
             "plastered", "bled", "motoring", "sing", "conflated", "troubled",
             "sized", "hopping", "tanned", "falling", "hissing", "fizzed",
             "failing", "filing", "happy", "sky", "relational", "conditional",
             "rational", "valency", "hesitancy", "digitizer", "conformably",
             "radically", "differently", "vilely", "analogousness",
             "vietnamization", "predication", "operator", "feudalism",
             "decisiveness", "hopefulness", "callousness", "formality",
             "sensitivity", "sensibility", "triplicate", "formative",
             "formalize", "electricity", "electrical", "hopeful", "goodness",
             "revival", "allowance", "inference", "airliner", "gyroscopic",
             "adjustable", "defensible", "irritant", "replacement",
             "adjustment", "dependent", "adoption", "homologou", "communism",
             "activate", "angularity", "homologous", "effective", "bowdlerize",
             "probate", "rate", "cease", "controll", "roll", "aerodynamics",
             "experimental", "slipstream", "boundary", "viscosity"]
    for t in tests:
        print("%-16s -> %s" % (t, stem(t)))
