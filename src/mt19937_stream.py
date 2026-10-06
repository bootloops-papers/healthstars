# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Anthropic, PBC. Part of the healthstars replication package; see LICENSE and NOTICE beside this file.
# Created by Matthew D. Schwartz.

"""Uniform RNG streams implemented from the documented algorithms.

Mersenne twister MT19937 (Matsumoto & Nishimura 1998) with the seeding that the
SAS documentation specifies for PROC SURVEYSELECT SEED= (SAS/STAT User's Guide,
"The SURVEYSELECT Procedure", Random Number Generation; Sarle & Wicklin, SAS
Global Forum 2018 paper 1810-2018: for seeds not divisible by 8192 the MT1998
initialization path applies; CMS's seed 8675309 mod 8192 = 8173 != 0). An
independent implementation written from those documents and validated against
the published test vectors below.

VALIDATED against published outputs: MT19937 with the mt19937-1.c (1999) seeding
  mt[0] = seed; mt[i] = 69069 * mt[i-1] mod 2^32
and standard tempering reproduces BOTH published SAS uniform vectors
(seed 54321 -> 0.43223 0.59780 0.77860 0.17483 0.39415;
 seed 12345 -> 0.58330 0.99363 0.58789 0.85747 0.82469, Sarle-Wicklin Fig. 1)
at the printed 5-decimal precision. The 2002 init_genrand and the 1998 two-step
Knuth init both FAIL these vectors.

The published vectors do not fix the uint32 -> double conversion at fine
precision: y/2^32, (y+0.5)/2^32, y/(2^32-1), (y+1)/(2^32+1) all agree at 5 dp
on them. For GROUPS= assignment only the RELATIVE ORDER / subinterval of
uniforms matters at scale ~1/N, so any order-preserving candidate is equivalent
for our purpose unless a uniform falls within ~1e-9 of a quota boundary — the
caller flags those events. This module's default is y * 2**-32; the record's
resampled replay adopts the conversion stated in groups_assign.frontswap as
part of its stated assumption (README, 'Assumed readout').

RANUNI (legacy, NOT used by CMS's call, kept for completeness/testing):
prime-modulus multiplicative congruential generator, modulus 2^31-1, multiplier
397204094 (Fishman & Moore 1982; SAS Language Reference). u = x / (2^31-1).

Paper: "Stars Misaligned: Medicare Star Ratings And Exact Clustering" (M. D. Schwartz, 2026), Section 2.4 and Appendix C.3
    (the Mersenne-twister stream reproduces the documented test vectors for the
    printed seed).
Run:  cd src && python3 mt19937_stream.py   (published-vector self-test; imported by groups_assign.py)
Requires: Python >= 3.10, standard library only.
"""

import sys

N624, M397 = 624, 397
MATRIX_A = 0x9908B0DF
UPPER, LOWER = 0x80000000, 0x7FFFFFFF
MASK32 = 0xFFFFFFFF


class SasMT:
    """MT19937 with the documented MT1998 (mt19937-1.c) seeding; yields the
    RAND('UNIFORM') stream as documented."""

    def __init__(self, seed, conversion="y/2^32"):
        if not (0 < seed < 2**31):
            raise ValueError("SAS SEED= must be a positive integer < 2^31")
        if seed % 8192 == 0:
            raise ValueError(
                f"seed {seed} divisible by 8192: SAS 9.4M3+ (MTHYBRID) would use "
                "the MT2002 init path here — not implemented, and not needed for 8675309")
        mt = [seed & MASK32]
        for i in range(1, N624):
            mt.append((69069 * mt[i - 1]) & MASK32)
        self.mt = mt
        self.mti = N624
        self.conversion = conversion

    def next_u32(self):
        if self.mti >= N624:
            mt = self.mt
            for k in range(N624):
                y = (mt[k] & UPPER) | (mt[(k + 1) % N624] & LOWER)
                mt[k] = mt[(k + M397) % N624] ^ (y >> 1) ^ (MATRIX_A if y & 1 else 0)
            self.mti = 0
        y = self.mt[self.mti]
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & MASK32

    def uniform(self):
        y = self.next_u32()
        c = self.conversion
        if c == "y/2^32":
            return y * 2.0**-32
        if c == "(y+0.5)/2^32":
            return (y + 0.5) * 2.0**-32
        if c == "y/(2^32-1)":
            return y / (2**32 - 1)
        if c == "(y+1)/(2^32+1)":
            return (y + 1) / (2**32 + 1)
        raise ValueError(c)


class SasRanuni:
    """Legacy RANUNI stream (not used by the CMS SURVEYSELECT call; reference only)."""

    MULT, MOD = 397204094, 2**31 - 1

    def __init__(self, seed):
        if not (0 < seed < self.MOD):
            raise ValueError("RANUNI seed must be in (0, 2^31-1)")
        self.x = seed

    def uniform(self):
        self.x = (self.MULT * self.x) % self.MOD
        return self.x / self.MOD


if __name__ == "__main__" and ("-h" in sys.argv[1:] or "--help" in sys.argv[1:]):
    print(__doc__)
    sys.exit(0)

if __name__ == "__main__":
    # Published SAS vectors, Sarle & Wicklin SGF 2018 paper 1810-2018, Figure 1.
    targets = {54321: [0.43223, 0.59780, 0.77860, 0.17483, 0.39415],
               12345: [0.58330, 0.99363, 0.58789, 0.85747, 0.82469]}
    for seed, tgt in targets.items():
        g = SasMT(seed)
        vals = [g.uniform() for _ in range(5)]
        assert all(abs(v - t) < 5e-6 for v, t in zip(vals, tgt)), (seed, vals)
    # conversion candidates preserve order (sanity: same u32 stream)
    g1, g2 = SasMT(8675309, "y/2^32"), SasMT(8675309, "(y+0.5)/2^32")
    a = [g1.uniform() for _ in range(1000)]
    b = [g2.uniform() for _ in range(1000)]
    assert sorted(range(1000), key=a.__getitem__) == sorted(range(1000), key=b.__getitem__)
    # RANUNI states from seed 8675309 (self-consistency values of this implementation)
    r = SasRanuni(8675309)
    xs = []
    for _ in range(3):
        r.uniform()
        xs.append(r.x)
    assert xs == [959153317, 785305884, 861217578], xs
    print("mt19937_stream: SAS published-vector tests passed; RANUNI states verified")
