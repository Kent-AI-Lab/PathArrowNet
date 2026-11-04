#!/bin/bash
: "${KEGG_HOME:=$(d=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); while [ "$d" != / ] && [ ! -e "$d/.kegg_release_root" ]; do d=$(dirname "$d"); done; echo "$d")}"  # release: repo root
# Downloads raw files for downstream tasks (resumable). Usage: bash download_all.sh
set -u
D=${KEGG_HOME}/pathway/expansion/downstream/data
dl(){ curl -sSL --retry 5 -C - --max-time 7200 -o "$1" "$2" && echo "OK $1" || echo "FAIL $1 $2"; }
dl $D/common/raw/hgnc_complete_set.txt https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
G=$D/coexpression_gtex/raw
dl $G/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
dl $G/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt
T=$D/tcga/raw; X=https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download
for f in Subtype_Immune_Model_Based.txt.gz TCGASubtype.20170308.tsv.gz TCGA_phenotype_denseDataOnlyDownload.tsv.gz Survival_SupplementalTable_S1_20171025_xena_sp; do dl $T/$f "$X/$f"; done
dl $T/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz "$X/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
dl $G/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz
