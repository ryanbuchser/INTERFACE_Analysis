#!/usr/bin/env python3
"""
design_utr_tiling_probes.py - tile short (9-15 nt) RNA probes along UTRs,
anchored on user-defined binding sites, trading off hybridization strength
against genome specificity.

INPUTS
------
--utrs     CSV/TSV with a header: gene, sequence [, utr_id | type]
             gene        gene name
             sequence    UTR sequence, 5'->3', sense (DNA or RNA alphabet)
             utr_id/type optional; needed only if a gene has several UTRs
                         (e.g. type = 5UTR / 3UTR -> id "geneX_5UTR")
--sites    CSV/TSV with a header: gene, start, stop [, label] [, utr]
             start/stop  1-based, inclusive, RELATIVE TO THE UTR SEQUENCE
             utr         optional UTR id / type when a gene has several UTRs
--genome   genome FASTA (DNA, .gz ok) used for the specificity search

HOW PROBES ARE CHOSEN
---------------------
1. Tile grid.  Around the centre of every binding site a grid of probe "slots"
   is laid out at --pitch nt spacing: slot 0 is centred on the site, slots
   +-1, +-2 ... go outward toward the 3'/5' end of the UTR.  By default the
   whole UTR is tiled (--tiles-per-side 0); set N > 0 to stop N tiles from
   each site.  Where two sites compete for the same stretch of UTR, each
   position goes to the nearest site.  A UTR with no listed sites is tiled
   from its centre.  EVERY probe is then labelled in the output: at_binding_site
   (does it overlap any listed site), overlapping_sites, overlap_with_site_nt,
   covers_whole_site, nearest_site and dist_to_nearest_site_nt.
2. Candidates.  For every slot, every probe of 9-15 nt whose centre lies within
   +-shift of the slot centre (--center-tol for slot 0, --slot-shift otherwise)
   is a candidate.  Probes are the exact reverse complement of the UTR.
3. Filters: GC%, homopolymers, nearest-neighbour duplex dG (Xia 1998 RNA/RNA),
   probe hairpin / self-dimer (folding backend).
4. SPECIFICITY (BLAST).  Each candidate's target sequence is searched against
   the genome with blastn-short (both strands, ungapped).  The on-target locus
   is removed by its exact diagonal, and every other alignment is re-scored
   over the full probe length.  specificity margin = probe length - best
   off-target matches (nt), capped at --spec-cap.  Candidates with margin
   < --min-spec-margin are dropped (--allow-nonunique keeps and penalises).
   --spec-crosscheck also runs an exhaustive k-mer mismatch search and keeps
   the stricter of the two (recommended for probes <= 12 nt, where BLAST's
   E-value / seed heuristics can miss weak partial matches).
5. HYBRIDIZATION STRENGTH (NUPACK).  For each candidate:
       dG_bind = G(window + probe) - G(window) - G(probe)
   from NUPACK partition-function free energies, where "window" is the target
   region plus --flank nt of UTR each side.  Target secondary structure
   therefore costs binding free energy automatically.  strength = -dG_bind.
   Also reported: mean unpaired probability of the target (NUPACK `pairs`).
6. Each slot gets the candidate with the best weighted score of normalised
   strength and specificity (--w-hyb / --w-spec); the slot's Pareto-optimal
   candidates (strength vs specificity) are flagged in the output.  Chosen
   neighbours may overlap by at most --max-pairwise-overlap nt.

BACKENDS
--------
--fold-backend auto|nupack|vienna|none   (auto: NUPACK, else ViennaRNA, else none)
--spec-backend auto|blast|internal       (auto: BLAST+ on PATH, else internal)

OUTPUT   <prefix>.probes.csv, <prefix>.probes.fasta [, <prefix>.all_candidates.csv]

EXAMPLE
-------
python design_utr_tiling_probes.py --genome genome.fa --utrs utrs.csv \
    --sites sites.csv --out-prefix out --spec-crosscheck
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr)


def info(msg):
    print(msg, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Sequence helpers
# --------------------------------------------------------------------------- #
_COMP = str.maketrans("ACGUTacgut", "UGCAAugcaa")
_DNA_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
_ENC = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
_DNA_PAIR = {"A": "T", "C": "G", "G": "C", "T": "A"}


def to_rna(s):
    return s.upper().replace("T", "U")


def to_dna(s):
    return s.upper().replace("U", "T")


def revcomp(s):
    return s.translate(_COMP)[::-1]


def revcomp_dna(s):
    return s.translate(_DNA_COMP)[::-1]


def encode(s):
    v = 0
    for ch in s:
        v = (v << 2) | _ENC[ch]
    return v


def gc_fraction(s):
    return (s.count("G") + s.count("C")) / len(s)


def max_homopolymer(s):
    return max(len(m.group(0)) for m in re.finditer(r"(.)\1*", s))


# --------------------------------------------------------------------------- #
# RNA/RNA nearest-neighbour thermodynamics (Xia et al. 1998)
# --------------------------------------------------------------------------- #
_NN = {
    "AA": (-6.82, -19.0), "AU": (-9.38, -26.7), "UA": (-7.69, -20.5),
    "CU": (-10.48, -27.1), "CA": (-10.44, -26.9), "GU": (-11.40, -29.5),
    "GA": (-12.44, -32.5), "CG": (-10.64, -26.7), "GG": (-13.39, -32.7),
    "GC": (-14.88, -36.9),
}
_INIT = (3.61, -1.5)
_TERM_AU = (3.72, 10.5)
R_CAL = 1.987
NN16 = {}
for _a in "ACGU":
    for _b in "ACGU":
        _dn = _a + _b
        NN16[_dn] = _NN[_dn] if _dn in _NN else _NN[revcomp(_dn)]


def duplex_thermo(target, temp_c, na, ct):
    """(dG, dH, dS, Tm_C) of the probe:target duplex (target given 5'->3')."""
    dH, dS = _INIT
    for x, y in zip(target, target[1:]):
        h, s = NN16[x + y]
        dH += h
        dS += s
    for end in (target[0], target[-1]):
        if end in "AU":
            dH += _TERM_AU[0]
            dS += _TERM_AU[1]
    if abs(na - 1.0) > 1e-9:
        dS += 0.368 * (len(target) - 1) * math.log(na)
    T = temp_c + 273.15
    dG = dH - T * dS / 1000.0
    tm = dH * 1000.0 / (dS + R_CAL * math.log(ct / 4.0)) - 273.15
    return dG, dH, dS, tm


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def _read_table(path):
    with open(path, newline="") as fh:
        head = fh.readline()
        fh.seek(0)
        delim = "\t" if head.count("\t") > head.count(",") else ","
        rdr = csv.DictReader(fh, delimiter=delim)
        cols = {c.strip().lower(): c for c in (rdr.fieldnames or [])}
        rows = [{k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
                for r in rdr]
    return cols, rows


def _pick(cols, *names):
    for n in names:
        if n in cols:
            return n
    return None


@dataclass
class Site:
    label: str
    start: int            # 0-based inclusive, UTR-relative
    end: int              # exclusive
    real: bool = True     # False = internal anchor for a UTR with no binding sites

    @property
    def center(self):
        return (self.start + self.end) / 2.0


@dataclass
class Utr:
    id: str
    gene: str
    kind: str
    seq: str                           # sense RNA
    sites: list = field(default_factory=list)
    slots: list = field(default_factory=list)
    abs_start: int | None = None       # on-target locus (forward coordinates)
    abs_end: int | None = None
    strand: str = "+"
    contig: str = ""
    g0: int | None = None
    g1: int | None = None


def read_utr_table(path):
    cols, rows = _read_table(path)
    g = _pick(cols, "gene", "gene_id", "locus_tag", "name")
    s = _pick(cols, "sequence", "seq", "utr", "utr_seq", "utr_sequence")
    if not (g and s):
        sys.exit(f"ERROR: {path} needs columns gene and sequence (found {list(cols)})")
    uid = _pick(cols, "utr_id", "id")
    kind = _pick(cols, "type", "kind", "utr_type")
    utrs, seen = [], Counter()
    for r in rows:
        if not r[g] or not r[s]:
            continue
        seq = to_rna(re.sub(r"\s+", "", r[s]))
        k = r[kind] if kind else ""
        i = r[uid] if uid and r[uid] else (f"{r[g]}_{k}" if k else r[g])
        seen[i] += 1
        if seen[i] > 1:
            i = f"{i}_{seen[i]}"
        bad = set(seq) - set("ACGU")
        if bad:
            warn(f"{i}: non-ACGU characters {sorted(bad)} - candidates covering them are skipped")
        utrs.append(Utr(i, r[g], k, seq))
    if not utrs:
        sys.exit("ERROR: no UTRs read")
    return utrs


def read_site_rows(path):
    cols, rows = _read_table(path)
    g = _pick(cols, "gene", "gene_id", "locus_tag", "name", "target")
    s = _pick(cols, "start", "begin", "from", "site_start")
    e = _pick(cols, "stop", "end", "to", "site_end", "site_stop")
    lab = _pick(cols, "label", "site", "site_id")
    hint = _pick(cols, "utr", "utr_id", "type", "kind")
    if not (g and s and e):
        sys.exit(f"ERROR: {path} needs columns gene, start, stop (found {list(cols)})")
    out = []
    for r in rows:
        if not r[g]:
            continue
        a, b = int(float(r[s])), int(float(r[e]))
        out.append((r[g], min(a, b), max(a, b), r[lab] if lab else "",
                    r[hint] if hint else ""))
    return out


def assign_sites(utrs, rows):
    by_id = {u.id.lower(): u for u in utrs}
    by_gene = defaultdict(list)
    for u in utrs:
        by_gene[u.gene.lower()].append(u)
    for gene, a, b, label, hint in rows:
        cands = []
        if hint:
            h = hint.lower()
            u = by_id.get(h) or by_id.get(f"{gene}_{hint}".lower())
            cands = [u] if u else [x for x in by_gene.get(gene.lower(), [])
                                   if x.kind.lower() == h]
        else:
            cands = by_gene.get(gene.lower()) or ([by_id[gene.lower()]]
                                                  if gene.lower() in by_id else [])
        if not cands:
            warn(f"site {gene}:{a}-{b} matches no UTR - skipped")
            continue
        fit = [u for u in cands if a <= len(u.seq)]
        if len(cands) > 1:
            warn(f"site {gene}:{a}-{b} is ambiguous between {[u.id for u in cands]}; "
                 f"using the first it fits (add a 'utr' column to the sites table)")
        if not fit:
            warn(f"site {gene}:{a}-{b} lies outside {cands[0].id} "
                 f"(length {len(cands[0].seq)}) - skipped")
            continue
        u = fit[0]
        s0, e0 = a - 1, min(b, len(u.seq))
        u.sites.append(Site(label or f"site{len(u.sites) + 1}", s0, e0))


# --------------------------------------------------------------------------- #
# Genome index (locus lookup + exhaustive k-mer cross-check)
# --------------------------------------------------------------------------- #
class GenomeIndex:
    def __init__(self, contigs):
        self.names, self.starts, parts, off = [], [], [], 0
        for name, seq in contigs.items():
            self.names.append(name)
            self.starts.append(off)
            parts.append(to_dna(seq))
            off += len(seq) + 1
        self.offsets = dict(zip(self.names, self.starts))
        self.F = "N".join(parts)
        self.N = len(self.F)
        self.rev_off = self.N + 1
        lut = np.full(256, 4, np.uint8)
        for i, b in enumerate(b"ACGT"):
            lut[b] = i
        fwd = lut[np.frombuffer(self.F.encode(), np.uint8)]
        rev = np.where(fwd < 4, 3 - fwd, 4).astype(np.uint8)[::-1]
        self.arr = np.concatenate([fwd, np.array([4], np.uint8), rev])
        self._tables = {}

    def contig_of(self, abs_pos):
        i = bisect.bisect_right(self.starts, abs_pos) - 1
        return self.names[i], abs_pos - self.starts[i]

    def locate(self, seq_rna):
        s = to_dna(seq_rna)
        p = self.F.find(s)
        if p >= 0:
            dup = self.F.find(s, p + 1) >= 0 or self.F.find(revcomp_dna(s)) >= 0
            return p, "+", dup
        q = self.F.find(revcomp_dna(s))
        if q >= 0:
            dup = self.F.find(revcomp_dna(s), q + 1) >= 0
            return q, "-", dup
        return None

    def kmer_table(self, k):
        if k in self._tables:
            return self._tables[k]
        a = self.arr
        n = len(a) - k + 1
        codes = np.zeros(n, np.int64)
        bad = np.zeros(n, bool)
        for j in range(k):
            seg = a[j:j + n]
            bad |= seg > 3
            codes = (codes << 2) | (seg.astype(np.int64) & 3)
        valid = np.flatnonzero(~bad)
        c = codes[valid]
        order = np.argsort(c, kind="stable")
        self._tables[k] = (c[order], valid[order])
        return self._tables[k]


def make_masks(k, d):
    out = []
    for pos in itertools.combinations(range(k), d):
        for alts in itertools.product((1, 2, 3), repeat=d):
            m = 0
            for p, al in zip(pos, alts):
                m |= al << (2 * p)
            out.append(m)
    return np.array(out, dtype=np.int64)


def locate_utrs(utrs, idx):
    for u in utrs:
        loc = idx.locate(u.seq)
        if loc is None:
            warn(f"{u.id}: UTR sequence not found verbatim in the genome - the "
                 f"on-target locus is unknown, so one perfect-match hit per probe "
                 f"will be treated as on-target")
            continue
        u.abs_start, u.strand, dup = loc
        u.abs_end = u.abs_start + len(u.seq)
        u.contig, u.g0 = idx.contig_of(u.abs_start)
        u.g1 = u.g0 + len(u.seq)
        if u.strand == "-":
            # abs_start was found on the reverse complement; convert to forward coords
            pass
        if dup:
            warn(f"{u.id}: the full UTR occurs more than once in the genome - "
                 f"probes will not be unique (consider --allow-nonunique)")


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
@dataclass
class Slot:
    site_idx: int
    j: int
    center: float
    shift: int
    cands: list = field(default_factory=list)
    chosen: object = None


@dataclass
class Cand:
    utr: str
    start: int
    k: int
    target: str
    probe: str
    gc: float
    dG_nn: float
    tm: float
    slots: list = field(default_factory=list)
    off_best: int | None = None
    margin: float | None = None
    n_close: int = 0
    hairpin: float | None = None
    selfdimer: float | None = None
    dG_bind: float | None = None
    access: float | None = None
    H: float = 0.0
    S: float = 0.0
    score: float = 0.0
    pareto: dict = field(default_factory=dict)
    probe_id: str = ""


def build_slots(u, a):
    L = len(u.seq)
    centers = [s.center for s in u.sites]
    slots = []
    for si, c in enumerate(centers):
        jmax = a.tiles_per_side if a.tiles_per_side > 0 else math.ceil(L / a.pitch)
        for j in range(-jmax, jmax + 1):
            cj = c + j * a.pitch
            if cj < a.min_len / 2.0 or cj > L - a.min_len / 2.0:
                continue
            d = abs(cj - c)
            lost = False
            for sj, c2 in enumerate(centers):
                if sj == si:
                    continue
                d2 = abs(cj - c2)
                if d2 < d - 1e-9 or (abs(d2 - d) <= 1e-9 and sj < si):
                    lost = True
                    break
            if not lost:
                slots.append(Slot(si, j, cj, a.center_tol if (j == 0 and u.sites[si].real) else a.slot_shift))
    return slots


def build_candidates(u, a, counters):
    L = len(u.seq)
    u.slots = build_slots(u, a)
    cmap = {}
    for sl in u.slots:
        for k in range(a.min_len, a.max_len + 1):
            lo = math.ceil(sl.center - k / 2.0 - sl.shift)
            hi = math.floor(sl.center - k / 2.0 + sl.shift)
            for i in range(lo, hi + 1):
                if i < 0 or i + k > L:
                    continue
                key = (i, k)
                if key not in cmap:
                    cmap[key] = make_candidate(u, i, k, a, counters)
                c = cmap[key]
                if c:
                    c.slots.append(sl)
                    sl.cands.append(c)
    return [c for c in cmap.values() if c]


def make_candidate(u, i, k, a, counters):
    target = u.seq[i:i + k]
    if set(target) - set("ACGU"):
        counters["ambiguous base"] += 1
        return None
    gc = gc_fraction(target)
    if not (a.min_gc <= gc <= a.max_gc):
        counters["GC out of range"] += 1
        return None
    if max_homopolymer(target) > a.max_homopolymer:
        counters["homopolymer run"] += 1
        return None
    dG, dH, dS, tm = duplex_thermo(target, a.temp, a.na, a.probe_conc + a.target_conc)
    if dG > a.min_dg:
        counters["duplex dG too weak"] += 1
        return None
    return Cand(u.id, i, k, target, revcomp(target), gc, dG, tm)


def own_interval(u, c):
    """1-based plus-strand (start1, end1) of the candidate's on-target locus."""
    if u.g0 is None:
        return None
    if u.strand == "+":
        s0 = u.g0 + c.start
    else:
        s0 = u.g1 - c.start - c.k
    return s0 + 1, s0 + c.k


# --------------------------------------------------------------------------- #
# Specificity: BLAST+ and exhaustive internal search
# --------------------------------------------------------------------------- #
def diag_matches(cseq, plus, const, q):
    """Matches of query q along a whole diagonal of a contig (1-based s)."""
    n, m = len(cseq), 0
    for pos in range(1, len(q) + 1):
        s = const + pos if plus else const - pos
        if 1 <= s <= n:
            b = cseq[s - 1]
            if (b == q[pos - 1]) if plus else (_DNA_PAIR.get(b) == q[pos - 1]):
                m += 1
    return m


def spec_blast(cands, utrs_by_id, contigs_dna, a, workdir):
    for exe in ("makeblastdb", a.blastn):
        if not shutil.which(exe):
            sys.exit(f"ERROR: '{exe}' not found on PATH (install BLAST+ or use --spec-backend internal)")
    gfa = os.path.join(workdir, "genome.fa")
    with open(gfa, "w") as fh:
        for n, s in contigs_dna.items():
            fh.write(f">{n}\n{s}\n")
    db = os.path.join(workdir, "genomedb")
    r = subprocess.run(["makeblastdb", "-in", gfa, "-dbtype", "nucl", "-out", db],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"ERROR: makeblastdb failed:\n{r.stderr}")
    qfa = os.path.join(workdir, "q.fa")
    with open(qfa, "w") as fh:
        for n, c in enumerate(cands):
            fh.write(f">c{n}\n{to_dna(c.target)}\n")
    ws = a.blast_word_size
    if ws == 0:
        ws = max(4, math.ceil((a.min_len - (a.min_spec_margin - 1)) / a.min_spec_margin))
    cmd = [a.blastn, "-task", "blastn-short", "-query", qfa, "-db", db,
           "-word_size", str(ws), "-evalue", str(a.blast_evalue), "-dust", "no",
           "-soft_masking", "false", "-strand", "both", "-ungapped",
           "-num_threads", str(a.threads),
           "-outfmt", "6 qseqid sseqid qstart qend sstart send"]
    info(f"  BLAST: {' '.join(cmd[:1] + cmd[1:3])} ... word_size={ws} evalue={a.blast_evalue}")
    seen = [set() for _ in cands]
    best = [-1] * len(cands)
    close = [0] * len(cands)
    skipped_self = [False] * len(cands)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        f = line.split("\t")
        if len(f) < 6:
            continue
        n = int(f[0][1:])
        c = cands[n]
        sid = f[1]
        qs, ss, se = int(f[2]), int(f[4]), int(f[5])
        plus = ss <= se
        const = ss - qs if plus else ss + qs
        key = (sid, plus, const)
        if key in seen[n]:
            continue
        seen[n].add(key)
        u = utrs_by_id[c.utr]
        own = own_interval(u, c)
        if own is not None and sid == u.contig:
            if u.strand == "+" and plus and const == own[0] - 1:
                continue
            if u.strand == "-" and (not plus) and const == own[1] + 1:
                continue
        m = diag_matches(contigs_dna[sid], plus, const, to_dna(c.target))
        if own is None and m == c.k and not skipped_self[n]:
            skipped_self[n] = True
            continue
        best[n] = max(best[n], m)
        if m >= c.k - (a.spec_cap - 1):
            close[n] += 1
    err = proc.stderr.read()
    if proc.wait():
        sys.exit(f"ERROR: blastn failed:\n{err}")
    for n, c in enumerate(cands):
        c.off_best = max(best[n], 0)
        c.margin = float(min(c.k - best[n], a.spec_cap)) if best[n] >= 0 else float(a.spec_cap)
        c.n_close = close[n]


def spec_internal(cands, utrs_by_id, idx, a, combine):
    by_k = defaultdict(list)
    for c in cands:
        by_k[c.k].append(c)
    dmax = a.spec_cap - 1
    for k, lst in sorted(by_k.items()):
        info(f"  exhaustive search: {len(lst)} candidate {k}-mers ...")
        sc, sp = idx.kmer_table(k)
        masks = {d: make_masks(k, d) for d in range(1, dmax + 1)}
        for c in lst:
            code = encode(c.target)
            counts = {}
            lo = int(np.searchsorted(sc, code, "left"))
            hi = int(np.searchsorted(sc, code, "right"))
            n0 = hi - lo
            u = utrs_by_id[c.utr]
            if n0:
                if u.g0 is None:
                    n0 -= 1
                else:
                    own = own_interval(u, c)
                    s_f = own[0] - 1 + idx.offsets[u.contig]
                    p_f = s_f if u.strand == "+" else idx.rev_off + (idx.N - s_f - k)
                    p_r = idx.rev_off + (idx.N - p_f - k) if p_f < idx.N else idx.N - (p_f - idx.rev_off) - k
                    n0 -= int(np.isin(sp[lo:hi], [p_f, p_r]).sum())
            counts[0] = max(n0, 0)
            for d in range(1, dmax + 1):
                nb = np.int64(code) ^ masks[d]
                counts[d] = int((np.searchsorted(sc, nb, "right")
                                 - np.searchsorted(sc, nb, "left")).sum())
            dmin = next((d for d in range(0, dmax + 1) if counts[d] > 0), None)
            margin = float(a.spec_cap if dmin is None else dmin)
            n_close = sum(counts.values())
            if combine and c.margin is not None:
                c.margin = min(c.margin, margin)
                c.n_close = max(c.n_close, n_close)
            else:
                c.margin, c.n_close = margin, n_close
            c.off_best = int(round(k - c.margin))


# --------------------------------------------------------------------------- #
# Folding backends: NUPACK (preferred) and ViennaRNA (fallback)
# --------------------------------------------------------------------------- #
class NupackBackend:
    name = "NUPACK"

    def __init__(self, temp, na, mg):
        import nupack  # noqa
        from nupack import Model, pairs, pfunc
        self._pairs, self._pfunc = pairs, pfunc
        self.model = Model(material="rna", celsius=temp, sodium=na, magnesium=mg)
        self._cache = {}
        self._smoke()

    def pf_energy(self, seqs):
        key = tuple(seqs)
        if key not in self._cache:
            res = self._pfunc(strands=list(seqs), model=self.model)
            if isinstance(res, (tuple, list)):
                g = res[1]
            else:
                g = getattr(res, "free_energy", res)
            self._cache[key] = float(g)
        return self._cache[key]

    def unpaired(self, seq):
        P = np.array(self._pairs(strands=[seq], model=self.model).to_array(), float)
        return np.clip(np.diag(P), 0.0, 1.0)

    def _smoke(self):
        t = "AUGGCUACGUAC"
        p = revcomp(t)
        d = self.pf_energy([t, p]) - self.pf_energy([t]) - self.pf_energy([p])
        if not d < -3:
            raise RuntimeError(f"NUPACK smoke test failed (duplex dG={d:.2f}); check API version")
        u = self.unpaired(t)
        if u.shape != (len(t),):
            raise RuntimeError("NUPACK pairs() output not understood")


class ViennaBackend:
    name = "ViennaRNA (fallback)"

    def __init__(self, temp, na, mg):
        import RNA
        self.RNA = RNA
        self.md = RNA.md()
        self.md.temperature = temp
        if hasattr(self.md, "salt"):
            self.md.salt = na
        self._cache = {}

    def pf_energy(self, seqs):
        key = tuple(seqs)
        if key not in self._cache:
            fc = self.RNA.fold_compound("&".join(seqs), self.md)
            _, mfe = fc.mfe()
            fc.exp_params_rescale(mfe)
            _, g = fc.pf()
            self._cache[key] = float(g)
        return self._cache[key]

    def unpaired(self, seq):
        fc = self.RNA.fold_compound(seq, self.md)
        _, mfe = fc.mfe()
        fc.exp_params_rescale(mfe)
        fc.pf()
        P = np.array(fc.bpp())[1:, 1:]
        return np.clip(1.0 - P.sum(axis=0) - P.sum(axis=1), 0.0, 1.0)


def make_fold_backend(a):
    order = {"auto": ["nupack", "vienna"], "nupack": ["nupack"],
             "vienna": ["vienna"], "none": []}[a.fold_backend]
    for name in order:
        try:
            cls = NupackBackend if name == "nupack" else ViennaBackend
            return cls(a.temp, a.na, a.mg)
        except Exception as e:  # noqa
            if a.fold_backend == name:
                sys.exit(f"ERROR: {name} backend unavailable: {e}")
            info(f"NOTE: {name} backend unavailable ({type(e).__name__}: {e})")
    return None


def accessibility_profile(fb, seq, window=200, stride=50):
    n = len(seq)
    prof = np.full(n, np.nan)
    best = np.full(n, -1.0)
    starts = [0] if n <= window else list(range(0, n - window + 1, stride))
    if n > window and starts[-1] + window < n:
        starts.append(n - window)
    for ws in starts:
        sub = seq[ws:ws + window]
        unp = fb.unpaired(sub)
        pos = np.arange(len(sub))
        dist = np.minimum(pos, len(sub) - 1 - pos)
        sl = slice(ws, ws + len(sub))
        better = dist > best[sl]
        prof[sl] = np.where(better, unp, prof[sl])
        best[sl] = np.where(better, dist, best[sl])
    return prof


# --------------------------------------------------------------------------- #
# Scoring and tiling selection
# --------------------------------------------------------------------------- #
def minmax(vals):
    lo, hi = min(vals), max(vals)
    return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in vals]


def score_utr(cs, a):
    if not cs:
        return
    hn = minmax([c.H for c in cs])
    sn = minmax([c.S for c in cs])
    wsum = a.w_hyb + a.w_spec
    for c, h, s in zip(cs, hn, sn):
        c.score = (a.w_hyb * h + a.w_spec * s) / wsum


def flag_pareto(sl, pool):
    for c in pool:
        dominated = any(d.H >= c.H and d.S >= c.S and (d.H > c.H or d.S > c.S)
                        for d in pool if d is not c)
        c.pareto[(sl.site_idx, sl.j)] = not dominated


def ov(c, d):
    return min(c.start + c.k, d.start + d.k) - max(c.start, d.start)


def select_tiles(u, a):
    chosen = []
    order = sorted(u.slots, key=lambda s: (abs(s.j), s.j, s.site_idx))
    for sl in order:
        pool = [c for c in sl.cands if getattr(c, "ok", True)]
        if not pool:
            warn(f"{u.id} / {u.sites[sl.site_idx].label} slot {sl.j:+d}: no candidate survived the filters")
            continue
        flag_pareto(sl, pool)
        pool.sort(key=lambda c: (-c.score, -c.H))
        for c in pool:
            if all(ov(c, d) <= a.max_pairwise_overlap for d in chosen):
                sl.chosen = c
                c.assigned = getattr(c, "assigned", None) or sl
                chosen.append(c)
                break
        else:
            warn(f"{u.id} / {u.sites[sl.site_idx].label} slot {sl.j:+d}: every candidate "
                 f"overlaps a chosen neighbour by > {a.max_pairwise_overlap} nt "
                 f"(raise --max-pairwise-overlap or --slot-shift)")
    return chosen


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Tile 9-15 nt RNA probes along UTRs, anchored on binding sites, "
                    "optimising hybridization strength (NUPACK) vs specificity (BLAST).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io = p.add_argument_group("input / output")
    io.add_argument("--genome", required=True)
    io.add_argument("--utrs", required=True, help="CSV/TSV: gene,sequence[,utr_id|type]")
    io.add_argument("--sites", required=True, help="CSV/TSV: gene,start,stop[,label][,utr]")
    io.add_argument("--out-prefix", default="utr_probes")
    io.add_argument("--all-candidates", action="store_true")

    t = p.add_argument_group("tiling")
    t.add_argument("--min-len", type=int, default=9)
    t.add_argument("--max-len", type=int, default=15)
    t.add_argument("--pitch", type=float, default=12.0, help="spacing between tile centres (nt)")
    t.add_argument("--tiles-per-side", type=int, default=0,
                   help="tiles each side of a site centre (0 = tile the whole UTR)")
    t.add_argument("--center-tol", type=int, default=1,
                   help="max offset (nt) of the site-centred probe's centre from the site centre")
    t.add_argument("--slot-shift", type=int, default=3,
                   help="max offset (nt) of an outer tile's centre from its grid position")
    t.add_argument("--max-pairwise-overlap", type=int, default=3)
    t.add_argument("--min-gc", type=float, default=0.30)
    t.add_argument("--max-gc", type=float, default=0.70)
    t.add_argument("--max-homopolymer", type=int, default=4)

    th = p.add_argument_group("thermodynamics / folding")
    th.add_argument("--fold-backend", choices=["auto", "nupack", "vienna", "none"], default="auto")
    th.add_argument("--temp", type=float, default=37.0)
    th.add_argument("--na", type=float, default=1.0, help="[Na+] (M)")
    th.add_argument("--mg", type=float, default=0.0, help="[Mg2+] (M)")
    th.add_argument("--probe-conc", type=float, default=1e-6)
    th.add_argument("--target-conc", type=float, default=1e-6)
    th.add_argument("--min-dg", type=float, default=-8.0,
                    help="pre-filter: nearest-neighbour duplex dG must be <= this")
    th.add_argument("--flank", type=int, default=25,
                    help="UTR nt each side of the target included when folding for dG_bind")
    th.add_argument("--max-hairpin-dg", type=float, default=-2.0,
                    help="drop probes whose own ensemble free energy is below this")
    th.add_argument("--max-selfdimer-dg", type=float, default=-6.0,
                    help="drop probes whose homodimer dG (G_dimer-2G_mono) is below this")

    sp = p.add_argument_group("specificity")
    sp.add_argument("--spec-backend", choices=["auto", "blast", "internal"], default="auto")
    sp.add_argument("--spec-crosscheck", action="store_true",
                    help="with BLAST, also run the exhaustive mismatch search; keep the stricter")
    sp.add_argument("--blastn", default="blastn")
    sp.add_argument("--blast-word-size", type=int, default=0, help="0 = auto (>=4)")
    sp.add_argument("--blast-evalue", type=float, default=1000)
    sp.add_argument("--threads", type=int, default=4)
    sp.add_argument("--spec-cap", type=int, default=3,
                    help="specificity margin (nt) saturates here")
    sp.add_argument("--min-spec-margin", type=int, default=2,
                    help="drop probes whose best off-target alignment has fewer than this many "
                         "mismatching/unaligned nt")
    sp.add_argument("--allow-nonunique", action="store_true")

    ob = p.add_argument_group("objective")
    ob.add_argument("--w-hyb", type=float, default=0.5, help="weight: hybridization strength")
    ob.add_argument("--w-spec", type=float, default=0.5, help="weight: specificity")
    return p.parse_args()


def main():
    a = parse_args()
    if a.min_spec_margin > a.spec_cap:
        a.spec_cap = a.min_spec_margin
    info("Reading inputs ...")
    contigs = {n: to_dna(s) for n, s in read_fasta(a.genome).items()}
    idx = GenomeIndex(contigs)
    utrs = read_utr_table(a.utrs)
    assign_sites(utrs, read_site_rows(a.sites))
    for u in utrs:
        if not u.sites:
            warn(f"{u.id}: no binding sites listed - tiling from the UTR centre; "
                 f"all its probes will be labelled as not at a binding site")
            mid = len(u.seq) // 2
            u.sites.append(Site("tile", mid, mid + 1, real=False))
    locate_utrs(utrs, idx)
    by_id = {u.id: u for u in utrs}

    fb = make_fold_backend(a)
    info(f"Folding backend: {fb.name if fb else 'none (using nearest-neighbour duplex dG only)'}")

    counters = Counter()
    cands = []
    for u in utrs:
        cands += build_candidates(u, a, counters)
    info(f"{len(cands)} candidates after sequence/NN filters (rejected: {dict(counters)})")
    if not cands:
        sys.exit("No candidates - relax --min-dg / GC limits.")

    # ---- specificity
    backend = a.spec_backend
    if backend == "auto":
        backend = "blast" if shutil.which(a.blastn) else "internal"
        if backend == "internal":
            info("NOTE: blastn not found on PATH -> using the exhaustive internal search "
                 "(install BLAST+ to use BLAST)")
    info(f"Specificity backend: {backend}" + (" + internal cross-check" if backend == "blast" and a.spec_crosscheck else ""))
    if backend == "blast":
        with tempfile.TemporaryDirectory() as wd:
            spec_blast(cands, by_id, contigs, a, wd)
        if a.spec_crosscheck:
            spec_internal(cands, by_id, idx, a, combine=True)
    else:
        spec_internal(cands, by_id, idx, a, combine=False)
    if not a.allow_nonunique:
        n0 = len(cands)
        keep = {id(c) for c in cands if c.margin >= a.min_spec_margin}
        cands = [c for c in cands if id(c) in keep]
        for u in utrs:
            for sl in u.slots:
                sl.cands = [c for c in sl.cands if id(c) in keep]
        info(f"{len(cands)}/{n0} candidates meet specificity margin >= {a.min_spec_margin} nt")
        if not cands:
            sys.exit("No specific candidates: try --allow-nonunique, a lower --min-spec-margin, "
                     "or longer probes (a 9-12-mer is rarely unique in a bacterial genome).")

    # ---- folding / hybridization strength
    if fb:
        info("Computing hybridization free energies ...")
        for u in utrs:
            mine = [c for c in cands if c.utr == u.id]
            prof = accessibility_profile(fb, u.seq)
            for c in mine:
                c.hairpin = fb.pf_energy([c.probe])
                c.selfdimer = fb.pf_energy([c.probe, c.probe]) - 2 * c.hairpin
                c.ok = c.hairpin >= a.max_hairpin_dg and c.selfdimer >= a.max_selfdimer_dg
                if not c.ok:
                    continue
                lo, hi = max(0, c.start - a.flank), min(len(u.seq), c.start + c.k + a.flank)
                win = u.seq[lo:hi]
                c.dG_bind = fb.pf_energy([win, c.probe]) - fb.pf_energy([win]) - c.hairpin
                c.access = float(np.nanmean(prof[c.start:c.start + c.k]))
                c.H = -c.dG_bind
    else:
        for c in cands:
            c.H = -c.dG_nn
    for c in cands:
        c.S = c.margin
    for u in utrs:
        score_utr([c for c in cands if c.utr == u.id and getattr(c, "ok", True)], a)

    # ---- select tiles
    selected = []
    for u in utrs:
        ch = select_tiles(u, a)
        for sl in u.slots:
            if sl.chosen:
                s = u.sites[sl.site_idx]
                tag = "t0" if sl.j == 0 else f"t{sl.j:+d}"
                sl.chosen.probe_id = f"{u.id}_{s.label}_{tag}"
                sl.chosen.site_slot = sl
        selected += ch

    # ---- output
    def row(c):
        u = by_id[c.utr]
        sl = getattr(c, "site_slot", None) or c.slots[0]
        anchor = u.sites[sl.site_idx]
        real = [x for x in u.sites if x.real]
        ovs = [(x, max(0, min(c.start + c.k, x.end) - max(c.start, x.start))) for x in real]
        hit = [(x, o) for x, o in ovs if o > 0]
        if hit:
            near, dist = max(hit, key=lambda t: t[1])[0], 0
        elif real:
            def gap(x):
                return max(x.start - (c.start + c.k), c.start - x.end, 0)
            near = min(real, key=gap)
            dist = gap(near)
        else:
            near, dist = None, ""
        f = lambda v, p=2: "" if v is None else round(v, p)
        return {
            "probe_id": c.probe_id, "utr": u.id, "gene": u.gene,
            "at_binding_site": bool(hit),
            "overlapping_sites": ";".join(x.label for x, _ in hit),
            "overlap_with_site_nt": max((o for _, o in hit), default=0),
            "nearest_site": near.label if near else "",
            "dist_to_nearest_site_nt": dist,
            "covers_whole_site": any(o == x.end - x.start for x, o in hit),
            "anchor_site": anchor.label if anchor.real else "", "tile_index": sl.j,
            "probe_5to3_RNA": c.probe, "target_5to3_RNA": c.target, "length": c.k,
            "target_start_utr_rel": c.start + 1, "target_end_utr_rel": c.start + c.k,
            "probe_center_minus_slot_center": round(c.start + c.k / 2.0 - sl.center, 1),
            "GC_pct": round(100 * c.gc, 1), "dG_duplex_NN_kcal": round(c.dG_nn, 2),
            "Tm_NN_C": round(c.tm, 1), "dG_bind_kcal": f(c.dG_bind),
            "hyb_strength_kcal": round(c.H, 2), "target_unpaired_prob": f(c.access),
            "spec_margin_nt": c.margin, "offtarget_best_matches_nt": c.off_best,
            "offtarget_close_hits": c.n_close, "probe_ensemble_dG": f(c.hairpin),
            "probe_selfdimer_dG": f(c.selfdimer),
            "pareto_optimal_in_slot": c.pareto.get((sl.site_idx, sl.j), ""),
            "score": round(c.score, 3),
        }

    rows = [row(c) for c in sorted(selected, key=lambda c: (c.utr, c.start))]
    csv_path, fa_path = f"{a.out_prefix}.probes.csv", f"{a.out_prefix}.probes.fasta"
    if rows:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with open(fa_path, "w") as fh:
            for r in rows:
                fh.write(f">{r['probe_id']} target={r['target_start_utr_rel']}-{r['target_end_utr_rel']} "
                         f"dGbind={r['dG_bind_kcal']} margin={r['spec_margin_nt']}\n{r['probe_5to3_RNA']}\n")
    if a.all_candidates:
        allc = [c for c in cands if getattr(c, "ok", True)]
        for c in allc:
            c.site_slot = c.slots[0]
        with open(f"{a.out_prefix}.all_candidates.csv", "w", newline="") as fh:
            ar = [row(c) for c in sorted(allc, key=lambda c: (c.utr, c.start, c.k))]
            w = csv.DictWriter(fh, fieldnames=list(ar[0]))
            w.writeheader()
            w.writerows(ar)
    info(f"\nWrote {len(rows)} probes -> {csv_path}, {fa_path}")


if __name__ == "__main__":
    main()