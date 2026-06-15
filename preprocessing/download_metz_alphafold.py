"""Download AlphaFoldDB structures for Metz targets via UniProt accessions."""

import argparse
import csv
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_XML = "https://rest.uniprot.org/uniprotkb/{accession}.xml"
AFDB_PDB = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb"
UNIPROT_NS = "{http://uniprot.org/uniprot}"
DOMAIN_KEYWORDS = (
    "protein kinase",
    "histidine kinase",
    "pi3k/pi4k",
    "pi3k",
    "pi4k",
    "pipk",
    "agc-kinase",
    "agc kinase",
    "cbs",
)


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


def get_uniprot_xml(accession):
    xml_text = request_text(UNIPROT_XML.format(accession=accession))
    return ET.fromstring(xml_text)


def parse_feature(feature, accession):
    if feature.tag != f"{UNIPROT_NS}feature":
        return None

    type_ = feature.attrib.get("type", "")
    description = feature.attrib.get("description", "")
    begin = ""
    end = ""
    position = ""

    for child in feature:
        if child.tag != f"{UNIPROT_NS}location":
            continue
        for loc in child:
            if loc.tag == f"{UNIPROT_NS}position":
                position = loc.attrib.get("position", "")
            elif loc.tag == f"{UNIPROT_NS}begin":
                begin = loc.attrib.get("position", "")
            elif loc.tag == f"{UNIPROT_NS}end":
                end = loc.attrib.get("position", "")

    if position and not begin and not end:
        begin = position
        end = position
    if not begin or not end:
        return None

    try:
        begin_i = int(begin)
        end_i = int(end)
    except ValueError:
        return None

    return {
        "type": type_,
        "description": description,
        "begin": begin_i,
        "end": end_i,
        "accession": accession,
    }


def parse_uniprot_features(root, accession):
    features = []
    for entry in root:
        for child in entry:
            parsed = parse_feature(child, accession)
            if parsed is not None:
                features.append(parsed)
    return features


def is_target_domain(feature):
    text = f"{feature['type']} {feature['description']}".lower()
    return any(keyword in text for keyword in DOMAIN_KEYWORDS)


def choose_domain_feature(features):
    candidates = [feature for feature in features if is_target_domain(feature)]
    if not candidates:
        return None

    def score(feature):
        text = f"{feature['type']} {feature['description']}".lower()
        length = feature["end"] - feature["begin"] + 1
        type_bonus = 1000 if feature["type"].lower() == "domain" else 0
        kinase_bonus = 500 if "kinase" in text else 0
        reasonable_length_bonus = 200 if 80 <= length <= 400 else 0
        return type_bonus + kinase_bonus + reasonable_length_bonus - abs(length - 250) / 1000

    return sorted(candidates, key=score, reverse=True)[0]


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


def crop_pdb_by_residue_range(input_path, output_path, begin, end):
    kept = 0
    with input_path.open("r", encoding="utf-8", errors="ignore") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        dst.write(f"REMARK Cropped UniProt residue range {begin}-{end}\n")
        for line in src:
            record = line[:6]
            if record in {"ATOM  ", "HETATM"}:
                try:
                    residue_id = int(line[22:26])
                except ValueError:
                    continue
                if begin <= residue_id <= end:
                    dst.write(line)
                    kept += 1
            elif record in {"MODEL ", "ENDMDL"}:
                dst.write(line)
        dst.write("END\n")
    return kept


def append_csv_row(path, fieldnames, row):
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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
    parser.add_argument("--no_crop", action="store_true", help="Download full AlphaFold PDB only")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    domain_dir = args.out_dir / "kinase_domains"
    if not args.no_crop:
        domain_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = args.out_dir / "metz_uniprot_mapping.csv"
    failed_path = args.out_dir / "download_failed.csv"

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
        "domain_status",
        "domain_type",
        "domain_description",
        "domain_begin",
        "domain_end",
        "domain_pdb_file",
        "domain_atom_count",
    ]
    failed_fieldnames = ["prot_id", "stage", "accession", "error"]
    processed_ids = set()
    if args.overwrite:
        mapping_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)
    elif mapping_path.exists() and mapping_path.stat().st_size > 0:
        with mapping_path.open(newline="", encoding="utf-8") as handle:
            processed_ids = {
                row["prot_id"]
                for row in csv.DictReader(handle)
                if row.get("prot_id")
            }

    mapped_count = 0
    failed_count = 0
    with args.metz_prots.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prot_id = row["prot_id"].strip()
            metz_seq = row["prot_seq"].strip()
            if prot_id in processed_ids:
                continue
            search_terms = [prot_id]
            alias = ALIASES.get(prot_id)
            if alias and alias != prot_id:
                search_terms.append(alias)

            entries = []
            for term in search_terms:
                try:
                    entries = query_uniprot(term)
                except (HTTPError, URLError, TimeoutError) as exc:
                    append_csv_row(
                        failed_path,
                        failed_fieldnames,
                        {"prot_id": prot_id, "stage": "uniprot", "accession": "", "error": str(exc)},
                    )
                    failed_count += 1
                    entries = []
                if entries:
                    break
                time.sleep(args.sleep)

            entry, status, identity = choose_entry(prot_id, metz_seq, entries)
            if entry is None:
                append_csv_row(
                    failed_path,
                    failed_fieldnames,
                    {"prot_id": prot_id, "stage": "mapping", "accession": "", "error": "no_uniprot_entry"},
                )
                failed_count += 1
                continue

            accession = entry["primaryAccession"]
            output_path = args.out_dir / f"{sanitize_filename(prot_id)}.pdb"
            domain_output_path = domain_dir / f"{sanitize_filename(prot_id)}_domain.pdb"
            download_status = "exists"
            if args.overwrite or not output_path.exists():
                try:
                    download_one(accession, output_path)
                    download_status = "downloaded"
                except (HTTPError, URLError, TimeoutError) as exc:
                    download_status = "failed"
                    append_csv_row(
                        failed_path,
                        failed_fieldnames,
                        {
                            "prot_id": prot_id,
                            "stage": "alphafold",
                            "accession": accession,
                            "error": str(exc),
                        },
                    )
                    failed_count += 1

            domain = None
            domain_status = "skipped" if args.no_crop else "not_found"
            domain_atom_count = ""
            if not args.no_crop:
                try:
                    root = get_uniprot_xml(accession)
                    domain = choose_domain_feature(parse_uniprot_features(root, accession))
                except (HTTPError, URLError, TimeoutError, ET.ParseError) as exc:
                    append_csv_row(
                        failed_path,
                        failed_fieldnames,
                        {
                            "prot_id": prot_id,
                            "stage": "uniprot_xml",
                            "accession": accession,
                            "error": str(exc),
                        },
                    )
                    failed_count += 1

                if domain is not None and output_path.exists():
                    if args.overwrite or not domain_output_path.exists():
                        kept = crop_pdb_by_residue_range(
                            input_path=output_path,
                            output_path=domain_output_path,
                            begin=domain["begin"],
                            end=domain["end"],
                        )
                        domain_atom_count = kept
                        domain_status = "cropped" if kept > 0 else "empty_crop"
                    else:
                        domain_status = "exists"

            append_csv_row(
                mapping_path,
                fieldnames,
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
                    "domain_status": domain_status,
                    "domain_type": domain["type"] if domain else "",
                    "domain_description": domain["description"] if domain else "",
                    "domain_begin": domain["begin"] if domain else "",
                    "domain_end": domain["end"] if domain else "",
                    "domain_pdb_file": domain_output_path.name if domain_output_path.exists() else "",
                    "domain_atom_count": domain_atom_count,
                },
            )
            mapped_count += 1
            domain_msg = ""
            if domain:
                domain_msg = f" domain={domain['description']}:{domain['begin']}-{domain['end']} {domain_status}"
            print(f"{prot_id}: {accession} {status} {download_status}{domain_msg}", flush=True)
            time.sleep(args.sleep)

    print(f"Saved mapping: {mapping_path}")
    print(f"Saved failures: {failed_path}")
    print(f"Mapped: {mapped_count}, failed records: {failed_count}")


if __name__ == "__main__":
    main()
