"""Parse cached human KGML files into node, relation and reaction tables.

Nothing here touches the network; all 372 KGML files come from the v2 audit
cache (kegg_feasibility_cache/kgml).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional

from .common import OLD_KGML, OLD_PATHWAY_LIST, log, log_failure

# Identical to KEGG_human_feasibility_v2.py so edge counts stay comparable.
# binding/association, dissociation, compound, missing interaction and the rare
# bare "indirect" label are deliberately NOT treated as directional.
DIRECTIONAL_SUBTYPES = {
    "activation", "inhibition", "expression", "repression", "indirect effect",
    "state change", "phosphorylation", "dephosphorylation", "glycosylation",
    "ubiquitination", "methylation",
}

ID_PREFIX_KIND = {
    "hsa": "gene", "ko": "ortholog", "cpd": "compound", "gl": "glycan",
    "dr": "drug", "path": "map", "rn": "reaction", "ec": "enzyme", "br": "brite",
}


def split_ids(name: str) -> List[str]:
    if not name or name == "undefined":
        return []
    return [t for t in name.split() if ":" in t]


def ids_with_prefix(ids: List[str], prefix: str) -> List[str]:
    p = prefix + ":"
    return [i for i in ids if i.startswith(p)]


def load_pathway_list() -> List[tuple]:
    rows = []
    for line in OLD_PATHWAY_LIST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pid, name = line.split("\t", 1)
            rows.append((pid.replace("path:", ""), name))
    return rows


def parse_brite_pathway_classes(text: Optional[str]) -> Dict[str, tuple]:
    """br08901 -> {'00010': (A class, B class)}."""
    out = {}
    if not text:
        return out
    a = b = ""
    for line in text.splitlines():
        if line.startswith("A"):
            a = re.sub(r"<[^>]+>", "", line[1:]).strip()
        elif line.startswith("B"):
            b = line[1:].strip()
        elif line.startswith("C"):
            parts = line[1:].strip().split(None, 1)
            if parts:
                out[parts[0]] = (a, b)
    return out


def parse_one(pid: str, listed_name: str, xml_text: str, classes: Dict[str, tuple]) -> dict:
    root = ET.fromstring(xml_text)
    title = root.attrib.get("title", "")
    num = root.attrib.get("number", pid[3:])
    cls_a, cls_b = classes.get(num, ("", ""))

    entries = {}
    nodes = []
    for e in root.findall("entry"):
        eid = e.attrib.get("id", "")
        etype = e.attrib.get("type", "")
        raw = e.attrib.get("name", "")
        ids = split_ids(raw)
        g = e.find("graphics")
        comps = [c.attrib.get("id", "") for c in e.findall("component")]
        rattr = [r.replace("rn:", "") for r in e.attrib.get("reaction", "").split() if r]
        ent = {
            "entry_id": eid, "entry_type": etype, "raw_name": raw, "ids": ids,
            "hsa": [i[4:] for i in ids_with_prefix(ids, "hsa")],
            "ko": [i[3:] for i in ids_with_prefix(ids, "ko")],
            "cpd": [i[4:] for i in ids_with_prefix(ids, "cpd")],
            "gl": [i[3:] for i in ids_with_prefix(ids, "gl")],
            "dr": [i[3:] for i in ids_with_prefix(ids, "dr")],
            "path": [i[5:] for i in ids_with_prefix(ids, "path")],
            "components": comps,
            "reaction_attr": rattr,
            "graphics_name": g.attrib.get("name", "") if g is not None else "",
        }
        entries[eid] = ent

    relations = []
    for idx, rel in enumerate(root.findall("relation")):
        subs = [(s.attrib.get("name", "").strip().lower(), s.attrib.get("value", ""))
                for s in rel.findall("subtype")]
        names = [s for s, _ in subs]
        relations.append({
            "relation_index": idx,
            "source_entry": rel.attrib.get("entry1", ""),
            "target_entry": rel.attrib.get("entry2", ""),
            "relation_type": rel.attrib.get("type", ""),
            "subtypes": names,
            "subtype_values": [v for _, v in subs],
            "directional": any(s in DIRECTIONAL_SUBTYPES for s in names),
        })

    reactions = []
    for idx, r in enumerate(root.findall("reaction")):
        subs = [(s.attrib.get("id", ""), s.attrib.get("name", "")) for s in r.findall("substrate")]
        prods = [(p.attrib.get("id", ""), p.attrib.get("name", "")) for p in r.findall("product")]
        reactions.append({
            "reaction_index": idx,
            "enzyme_entry": r.attrib.get("id", ""),
            "reaction_ids": [x.replace("rn:", "") for x in r.attrib.get("name", "").split()],
            "reaction_type": r.attrib.get("type", ""),
            "substrates": subs,
            "products": prods,
        })

    # Reaction elements whose id attribute points at an entry (the enzyme node).
    rxn_by_entry = defaultdict(set)
    for r in reactions:
        rxn_by_entry[r["enzyme_entry"]].update(r["reaction_ids"])
    for eid, ent in entries.items():
        ent["reaction_elements"] = sorted(rxn_by_entry.get(eid, ()))

    return {
        "pathway_id": pid, "listed_name": listed_name, "pathway_name": title,
        "class_a": cls_a, "class_b": cls_b,
        "entries": entries, "relations": relations, "reactions": reactions,
    }


def parse_all(classes: Dict[str, tuple]) -> Dict[str, dict]:
    parsed = {}
    for pid, listed in load_pathway_list():
        f = OLD_KGML / f"{pid}.xml"
        if not f.exists():
            log_failure("kgml", pid, "KGML missing from cache (not downloaded by v2 audit)")
            continue
        try:
            parsed[pid] = parse_one(pid, listed, f.read_text(encoding="utf-8"), classes)
        except Exception as e:
            log_failure("kgml", pid, f"parse error: {e}")
    log(f"parsed {len(parsed)} cached KGML pathways")
    return parsed


def expand_group_genes(entries: dict, eid: str) -> List[str]:
    """Human genes of a gene entry, or of all gene components of a group."""
    e = entries.get(eid)
    if not e:
        return []
    if e["entry_type"] == "gene":
        return e["hsa"]
    if e["entry_type"] == "group":
        out = []
        for c in e["components"]:
            ce = entries.get(c)
            if ce and ce["entry_type"] == "gene":
                out.extend(ce["hsa"])
        return sorted(set(out), key=int)
    return []
