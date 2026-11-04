#!/usr/bin/env python3
"""
Case study (Sec. 5.1, Suppl. S8): confident non-KEGG predictions of seed-averaged PathArrowNet on strict split 1 and their
automatic database support. The rules were fixed before any literature search (the selection was hashed before searching).

Rules
 Novel predictions (one per test source KO):
  1. candidate v = highest-scoring KO for source u, excluding special tokens, u itself, and every KO v such that u->v or
     v->u is a KEGG directed KO edge in ANY split (kegg_dataset/splits_ko/edge_split.csv = all 16,890 edges).
  2. keep only if v is ranked ABOVE every true test answer of u (the model's overall top-1 for u is a non-KEGG KO).
  3. confidence = within-row z-score of v's score (mean / sd over all real KO columns).
  4. redundancy: drop pairs where u or v maps to > 4 human genes (large families); at most one candidate per target KO
     and per unordered pair (highest z kept).
  5. STRING experimental max score of the KO pair (ko_background_exp150_full.tsv; <150 -> 0). Pairs with score >= 700
     are 'obvious' and listed separately (top 5), not part of the literature set.
  6. literature set = top 15 remaining by z.
 Success cases: test A queries whose true target is ranked 1 (filtered rank), STRING exp < 150 for the pair, at most one
  per pathway, top 5 by z of the target.
 Database flags (from expansion/extdb_eval/cache/edges*.tsv.gz, built by expansion/extdb_eval/build_edges*.py): x->y or y->x
  in SIGNOR global / SIGNOR pathways / WikiPathways / NCI-PID; Reactome next-step flag and any-KGML-relation flag.
 Base rate: database support of all filtered candidates against a random control (same source KO, random non-KEGG target KO
  with a human gene, 20 draws per source, seed 0).

    python evaluation/case_study.py              -> results/evaluation/case_study/{candidates.json,candidates.tsv,db_base_rate.json}
"""
import csv, gzip, json, random
from collections import defaultdict
import numpy as np
from common import OUT as _OUT, PAN_SA, PW as ROOT, SL

OUT = _OUT / "case_study"
voc = list(csv.DictReader((ROOT / "kegg_dataset/token_vocab_ko_full.csv").open()))
tok = {int(r["token_id"]): r["token"] for r in voc}
kid = {r["token"]: int(r["token_id"]) for r in voc}
kname = {r["token"]: r["ko_name"] for r in voc}
real = np.array([i for i, t in tok.items() if not t.startswith("[")])

genes = defaultdict(list)
sym = {}
for line in (ROOT / "kegg_dataset_cache/kegg_rest/list_hsa.tsv").read_text().splitlines():
    p = line.split("\t")
    if len(p) >= 4:
        sym[p[0]] = p[3].split(";")[0].split(",")[0].strip()
for line in (ROOT / "kegg_dataset_cache/kegg_rest/link_ko_hsa.tsv").read_text().splitlines():
    g, k = line.split()
    genes[k[3:]].append(sym.get(g, g))

edges = set()
for r in csv.DictReader((ROOT / "kegg_dataset/splits_ko/edge_split.csv").open()):
    edges.add((r["source_ko"], r["target_ko"]))
blocked = defaultdict(set)
for a, b in edges:
    blocked[a].add(b); blocked[b].add(a)

stringexp = {}
with (ROOT / "kegg_dataset_cache/string/ko_background_exp150_full.tsv").open() as f:
    for r in csv.DictReader(f, delimiter="\t"):
        stringexp[frozenset((r["ko_a"], r["ko_b"]))] = int(r["max_experimental_score"])

db = defaultdict(set); flags = defaultdict(dict)
for fn in ("edges.tsv.gz", "edges_pid.tsv.gz"):
    with gzip.open(ROOT / "expansion/extdb_eval/cache" / fn, "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            db[(r["x"], r["y"])].add(r["db"] + ("" if not r["rels"] else f"[{r['rels']}]"))
            fl = flags[frozenset((r["x"], r["y"]))]
            fl["reactome"] = max(fl.get("reactome", 0), int(r["reactome"] or 0))
            fl["kgml_any"] = max(fl.get("kgml_any", 0), int(r["kgml_any"] or 0))


def dbinfo(a, b):
    return {"fwd": sorted(db.get((a, b), [])), "rev": sorted(db.get((b, a), [])), **flags.get(frozenset((a, b)), {})}


qs = SL.load_queries("strict_v1_full", "test", "A")
S = np.load(SL.SCORES / "strict_v1_full" / PAN_SA / "s0" / "seq_test_A.npz")["scores"]
assert S.shape[0] == len(qs)
by_src = defaultdict(list)
for i, q in enumerate(qs):
    by_src[q["source"]].append(i)

novel, success = [], []
for u, idx in by_src.items():
    row = S[idx[0]].astype(np.float64)
    assert all(np.allclose(S[j], S[idx[0]]) for j in idx), "rows differ for same source"
    U = tok[u]
    mu, sd = row[real].mean(), row[real].std()
    answers = {qs[j]["target"] for j in idx}
    best_ans = max(row[t] for t in answers)
    order = real[np.argsort(-row[real])]
    for v in order[:200]:
        V = tok[int(v)]
        if int(v) == u or V in blocked[U]:
            continue
        if row[v] > best_ans:
            novel.append(dict(u=U, v=V, z=float((row[v] - mu) / sd), score=float(row[v])))
        break
    for j in idx:
        q = qs[j]; t = q["target"]
        sc = row.copy(); sc[[x for x in q["exclude"]]] = -np.inf
        rank = int((sc[real] > sc[t]).sum()) + 1
        if rank == 1:
            success.append(dict(u=U, v=tok[t], z=float((row[t] - mu) / sd), pathway=q["pathway_ids"] if "pathway_ids" in q else q.get("cluster")))


def enrich(c):
    c["string_exp"] = stringexp.get(frozenset((c["u"], c["v"])), 0)
    c["genes_u"], c["genes_v"] = genes.get(c["u"], []), genes.get(c["v"], [])
    c["name_u"], c["name_v"] = kname.get(c["u"], ""), kname.get(c["v"], "")
    c["db"] = dbinfo(c["u"], c["v"])
    return c


novel = [enrich(c) for c in novel]
novel = [c for c in novel if 0 < len(c["genes_u"]) <= 4 and 0 < len(c["genes_v"]) <= 4]
novel.sort(key=lambda c: -c["z"])
seen_v, seen_pair, kept = set(), set(), []
for c in novel:
    pr = frozenset((c["u"], c["v"]))
    if c["v"] in seen_v or pr in seen_pair:
        continue
    seen_v.add(c["v"]); seen_pair.add(pr); kept.append(c)
obvious = [c for c in kept if c["string_exp"] >= 700][:5]
lit = [c for c in kept if c["string_exp"] < 700][:15]
success = [enrich(c) for c in success]
success = [c for c in success if c["string_exp"] < 150]
success.sort(key=lambda c: -c["z"])
sp, succ = set(), []
for c in success:
    p = str(c["pathway"]).split(";")[0]
    if p in sp:
        continue
    sp.add(p); succ.append(c)
succ = succ[:5]
res = dict(n_test_sources=len(by_src), n_novel_top1=len(novel) , n_after_filters=len(kept),
           n_string700_after_filters=sum(c["string_exp"] >= 700 for c in kept),
           literature_set=lit, obvious_string700=obvious, success_cases=succ)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "candidates.json").write_text(json.dumps(res, indent=1))
with (OUT / "candidates.tsv").open("w") as f:
    f.write("set\tu\tv\tgenes_u\tgenes_v\tz\tstring_exp\tdb_fwd\tdb_rev\treactome\tkgml_any\n")
    for name, L in (("lit", lit), ("obvious", obvious), ("success", succ)):
        for c in L:
            f.write("\t".join([name, c["u"], c["v"], ",".join(c["genes_u"]), ",".join(c["genes_v"]), f"{c['z']:.2f}",
                               str(c["string_exp"]), ";".join(c["db"]["fwd"]), ";".join(c["db"]["rev"]),
                               str(c["db"].get("reactome", "")), str(c["db"].get("kgml_any", ""))]) + "\n")
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}))

# ---- automatic database support against a random control
has = lambda a, b: bool(db.get((a, b)) or db.get((b, a)))
fwd = lambda a, b: bool(db.get((a, b)))
pool = [tok[int(i)] for i in real if genes.get(tok[int(i)])]
rng = random.Random(0)
rnd = []
for c in kept:
    for _ in range(20):
        v = rng.choice(pool)
        while v == c["u"] or v in blocked[c["u"]]:
            v = rng.choice(pool)
        rnd.append((c["u"], v))
res = dict(n_pred=len(kept), pred_any=sum(has(c["u"], c["v"]) for c in kept), pred_fwd=sum(fwd(c["u"], c["v"]) for c in kept),
           n_rand=len(rnd), rand_any=sum(has(a, b) for a, b in rnd), rand_fwd=sum(fwd(a, b) for a, b in rnd),
           pred_string150=sum(c["string_exp"] >= 150 for c in kept))
lo = [c for c in kept if c["string_exp"] < 150]
res.update(n_pred_string_lt150=len(lo), pred_any_string_lt150=sum(has(c["u"], c["v"]) for c in lo))
(OUT / "db_base_rate.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
