from pathlib import Path

DATASET_ROOTS = {
    "val": Path("./data/dataset_for_val_ML"),
    "pretrain": Path("./data/dataset_for_pretrain"),
}

DEFAULT_PROJECTS_VAL = [
    "TCGA-KIRC", "TCGA-READ", "TCGA-ESCA", "TCGA-KICH", "TCGA-UCS",
    "TCGA-KIRP", "TCGA-ACC", "TCGA-SKCM", "TCGA-BLCA", "TCGA-LUSC"
]

OMICS_PREFIX = {
    "1": "mRNA",
    "2": "DNA-meth",
    "3": "miRNA",
}
