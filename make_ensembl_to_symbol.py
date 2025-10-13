import pandas as pd
from mygene import MyGeneInfo

# 读 1_featname.csv，取第一列，去版本号与重复
feat = pd.read_csv("1_featname.csv", header=None)[0].astype(str)
ensg = feat.str.replace(r"\..*$", "", regex=True).drop_duplicates().tolist()

mg = MyGeneInfo()
q = mg.querymany(ensg, scopes="ensembl.gene", fields="symbol", species="human")

rows = []
seen = set()
for r in q:
    ensg_id = r.get('query')
    sym = r.get('symbol', '') if not r.get('notfound', False) else ''
    if ensg_id and ensg_id not in seen:
        rows.append((ensg_id, sym))
        seen.add(ensg_id)

# 对缺失项补空
missing = [e for e in ensg if e not in seen]
rows += [(e, '') for e in missing]

pd.DataFrame(rows, columns=["ensembl_gene_id", "symbol"]).to_csv("ensembl_to_symbol.csv", index=False)
print("Saved ensembl_to_symbol.csv with", len(rows), "rows")
