#!/usr/bin/env python3
"""UniProt Swiss-Prot human: accession, primary gene symbol, sequence (paginated REST search, 500 per page)
-> kegg_dataset_cache/text/uniprot_human_sprot_seq.tsv"""
import re, time
from pathlib import Path
import requests
OUT = Path(__file__).resolve().parent.parent / "kegg_dataset_cache/text/uniprot_human_sprot_seq.tsv"
url = ("https://rest.uniprot.org/uniprotkb/search?fields=accession,gene_primary,sequence&format=tsv&size=500"
       "&query=(organism_id:9606)%20AND%20(reviewed:true)")
lines, header = [], None
while url:
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=120); r.raise_for_status(); break
        except Exception as e:
            print("retry", attempt, e, flush=True); time.sleep(10)
    rows = r.text.rstrip("\n").split("\n"); header = rows[0]; lines += rows[1:]
    m = re.search(r'<([^>]+)>; rel="next"', r.headers.get("Link", "")); url = m.group(1) if m else None
    print(len(lines), flush=True)
OUT.write_text(header + "\n" + "\n".join(lines) + "\n"); print("DONE", len(lines))
