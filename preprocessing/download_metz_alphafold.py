"""Download AlphaFoldDB structures for Metz targets via UniProt accessions."""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
AFDB_PDB = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb"


ALIASES = {
    "ACK1": "TNK2",
    "CDC2": "CDK1",
    "CDC2L6": "CDK11B",
    "CDC42BPA": "CDC42BPA",
    "DCAMKL1": "DCLK1",
    "GPRK5": "GRK5",
    "IKBKB": "IKBKB",
    "IKBKE": "IKBKE",
    "SGK": "SGK1",
    "STK6": "AURKA",
    "STK12": "AURKB",
    "STK22B": "TSSK2",
    "STK22D": "TSSK1B",
    "TAO1": "TAOK1",
    "TOPK": "PBK",
}


def request_text(url, timeout=60):
    request = Request(url, headers={"User-Agent": "HierDTA-metz-alphafold/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def request_bytes(url, timeout=120):
    headers = {"User-Agent": "HierDTA-metz-alphafold/1.0"}
    last_error = None
    for _ in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.content
            response.raise_for_status()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.0)
    raise URLError(str(last_error))


def sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def sequence_match_status(metz_seq, candidate_seq):
    if metz_seq == candidate_seq:
        return "exact", 1.0
    if metz_seq in candidate_seq:
        return "metz_subsequence", len(metz_seq) / max(len(candidate_seq), 1)
    if candidate_seq in metz_seq:
        return "uniprot_subsequence", len(candidate_seq) / max(len(metz_seq), 1)
    matches = sum(a == b for a, b in zip(metz_seq, candidate_seq))
    identity = matches / max(min(len(metz_seq), len(candidate_seq)), 1)
    return "length_or_sequence_mismatch", identity


def gene_names(entry):
    names = set()
    for gene in entry.get("genes", []):
        if gene.get("geneName", {}).get("value"):
            names.add(gene["geneName"]["value"].upper())
        for synonym in gene.get("synonyms", []):
            if synonym.get("value"):
                names.add(synonym["value"].upper())
    return names


def query_uniprot(term):
    fields = "accession,id,gene_names,protein_name,organism_name,length,sequence"
    query = f'(gene_exact:{term} OR gene:{term}) AND organism_id:9606 AND reviewed:true'
    url = (
        f"{UNIPROT_SEARCH}?query={quote(query)}&format=json&fields={fields}"
        "&size=10"
    )
    data = json.loads(request_text(url))
    return data.get("results", [])


def choose_entry(prot_id, metz_seq, entries):
    if not entries:
        return None, "not_found", 0.0

    target_names = {prot_id.upper(), ALIASES.get(prot_id, prot_id).upper()}
    scored = []
    for entry in entries:
        seq = entry.get("sequence", {}).get("value", "")
        status, identity = sequence_match_status(metz_seq, seq)
        names = gene_names(entry)
        gene_hit = bool(target_names & names)
        exact = status == "exact"
        contains = status in {"metz_subsequence", "uniprot_subsequence"}
        length_delta = abs(len(metz_seq) - len(seq))
        score = (
            int(exact) * 1000
            + int(contains) * 200
            + int(gene_hit) * 100
            + identity * 10
            - length_delta / 10000
        )
        scored.append((score, entry, status, identity))
    scored.sort(key=lambda item: item[0], reverse=True)
    _, entry, status, identity = scored[0]
    return entry, status, identity


def download_one(accession, output_path):
    content = request_bytes(AFDB_PDB.format(accession=accession))
    output_path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metz_prots",
        type=Path,
        default=PROJECT_ROOT / "data" / "metz" / "metz_prots.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "metz" / "prot_3D_for_Metz",
    )
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = args.out_dir / "metz_uniprot_mapping.csv"
    failed_path = args.out_dir / "download_failed.csv"

    rows = []
    failed = []
    with args.metz_prots.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prot_id = row["prot_id"].strip()
            metz_seq = row["prot_seq"].strip()
            search_terms = [prot_id]
            alias = ALIASES.get(prot_id)
            if alias and alias != prot_id:
                search_terms.append(alias)

            entries = []
            for term in search_terms:
                try:
                    entries = query_uniprot(term)
                except (HTTPError, URLError, TimeoutError) as exc:
                    failed.append({"prot_id": prot_id, "stage": "uniprot", "error": str(exc)})
                    entries = []
                if entries:
                    break
                time.sleep(args.sleep)

            entry, status, identity = choose_entry(prot_id, metz_seq, entries)
            if entry is None:
                failed.append({"prot_id": prot_id, "stage": "mapping", "error": "no_uniprot_entry"})
                continue

            accession = entry["primaryAccession"]
            output_path = args.out_dir / f"{sanitize_filename(prot_id)}.pdb"
            download_status = "exists"
            if args.overwrite or not output_path.exists():
                try:
                    download_one(accession, output_path)
                    download_status = "downloaded"
                except (HTTPError, URLError, TimeoutError) as exc:
                    download_status = "failed"
                    failed.append(
                        {
                            "prot_id": prot_id,
                            "stage": "alphafold",
                            "accession": accession,
                            "error": str(exc),
                        }
                    )

            rows.append(
                {
                    "prot_id": prot_id,
                    "uniprot_accession": accession,
                    "uniprot_id": entry.get("uniProtkbId", ""),
                    "uniprot_genes": ";".join(sorted(gene_names(entry))),
                    "metz_length": len(metz_seq),
                    "uniprot_length": entry.get("sequence", {}).get("length", ""),
                    "sequence_status": status,
                    "sequence_identity_simple": f"{identity:.6f}",
                    "pdb_file": output_path.name if output_path.exists() else "",
                    "download_status": download_status,
                }
            )
            print(f"{prot_id}: {accession} {status} {download_status}", flush=True)
            time.sleep(args.sleep)

    fieldnames = [
        "prot_id",
        "uniprot_accession",
        "uniprot_id",
        "uniprot_genes",
        "metz_length",
        "uniprot_length",
        "sequence_status",
        "sequence_identity_simple",
        "pdb_file",
        "download_status",
    ]
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    failed_fieldnames = ["prot_id", "stage", "accession", "error"]
    with failed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=failed_fieldnames)
        writer.writeheader()
        writer.writerows(failed)

    print(f"Saved mapping: {mapping_path}")
    print(f"Saved failures: {failed_path}")
    print(f"Mapped: {len(rows)}, failed records: {len(failed)}")


if __name__ == "__main__":
    main()
