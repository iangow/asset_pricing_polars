from db2pq import wrds_update_pq
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()

    tables = {
        "crsp": ["msf", "dsf", "msenames", "msedelist", "msi", "ccmxpf_lnkused"],
        "comp": ["funda"],
        "ff": ["factors_monthly", "factors_daily"],
    }

    for schema, names in tables.items():
        for name in names:
            wrds_update_pq(name, schema)
