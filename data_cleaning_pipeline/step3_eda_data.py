"""
Step 3 — EDA Data Preparation
Loads the deduplicated review data, then applies:
  - Timestamp → date conversion
  - Column drops
  - Category classification (laptop / tablet / desktop / other)
  - Brand extraction
  - Non-English review translation
  - Vader sentiment scoring
  - Price tier assignment

Data source (controlled by environment variable USE_LOCAL_CSV):
  LOCAL  (default): reads  dataset/digital_devices_reviews_no_duplicates.csv
                    writes dataset/eda_ready.csv
  CLOUD:            reads  BigQuery SOURCE_TABLE
                    writes BigQuery OUTPUT_TABLE

To run locally: 
    python pipeline/step3_eda_data.py

To force BigQuery mode:
    USE_LOCAL_CSV=false python pipeline/step3_eda_data.py
"""

import os
import re
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data on first run
for resource in ("vader_lexicon", "stopwords", "wordnet"):
    nltk.download(resource, quiet=True)

# Config 
PROJECT_ID   = os.environ.get("GCP_PROJECT", "cs163-project-487801")
SOURCE_TABLE = f"{PROJECT_ID}.amazon_digital_devices_cleaned.digital_devices_reviews_no_duplicates"
OUTPUT_TABLE = f"{PROJECT_ID}.amazon_digital_devices_cleaned.eda_ready"

# Default to local CSV — set USE_LOCAL_CSV=false to use BigQuery
USE_LOCAL_CSV = os.environ.get("USE_LOCAL_CSV", "true").lower() != "false"

LOCAL_INPUT_CSV  = os.path.join("dataset", "digital_devices_reviews_no_duplicates.csv")
LOCAL_OUTPUT_CSV = os.path.join("dataset", "eda_ready.csv")



#  Section 1: Load data

def load_reviews() -> pd.DataFrame:
    if USE_LOCAL_CSV:
        print(f"  Loading reviews from local CSV: {LOCAL_INPUT_CSV} …")
        if not os.path.exists(LOCAL_INPUT_CSV):
            raise FileNotFoundError(
                f"Local CSV not found at '{LOCAL_INPUT_CSV}'.\n"
                "Place your post-Step-2 export there, or set USE_LOCAL_CSV=false "
                "to load from BigQuery."
            )
        df = pd.read_csv(LOCAL_INPUT_CSV, low_memory=False)
    else:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID)
        query  = f"SELECT * FROM `{SOURCE_TABLE}`"
        print("  Loading reviews from BigQuery …")
        df = client.query(query).to_dataframe()

    print(f"  Loaded {len(df):,} rows")
    return df



#  Section 2: Basic cleaning

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Timestamp conversion, drop unused cols, filter Office Products."""
    df = df.copy()

    # Parse timestamp (milliseconds → datetime)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Drop columns not needed for EDA
    cols_to_drop = [c for c in ["timestamp", "pred_label", "rn"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Filter out Office Products (not a digital device category)
    df = df[df["main_category"] != "Office Products"].copy()

    # Add date parts for time-series analysis
    df["year"]         = df["date"].dt.year
    df["month"]        = df["date"].dt.month
    df["date_of_month"] = df["date"].dt.day
    df["day_of_week"]  = df["date"].dt.day_name()

    # Review length feature
    df["review_length"] = df["review_text"].str.len()

    # Price missing signal
    df["price_missing"] = df["price"].isna().astype(int)

    return df


#  Section 3: Price tiers

def assign_price_tiers(df: pd.DataFrame) -> pd.DataFrame:
    with_price = df[df["price_missing"] == 0]["price"]
    quantile   = with_price.quantile([0, 0.5, 0.75, 0.975, 1])

    def _tier(price):
        if pd.isna(price) or price == 0:
            return "Unknown"
        if price >= quantile.loc[0.975]:
            return "Premium"
        if price > quantile.loc[0.75]:
            return "High"
        if price > quantile.loc[0.5]:
            return "Medium"
        return "Low"

    df = df.copy()
    df["price_tier"] = df["price"].apply(_tier)
    return df



#  Section 4: COVID period assignment

def assign_covid_period(df: pd.DataFrame) -> pd.DataFrame:
    def _label(date):
        if pd.isna(date):
            return "unknown"
        if date < pd.Timestamp("2020-03-01"):
            return "pre-COVID"
        if date < pd.Timestamp("2021-07-01"):
            return "during-COVID"
        return "post-COVID"

    df = df.copy()
    df["covid_period"] = df["date"].apply(_label)
    return df


#  Section 5: Device category classification

# Products manually verified as belonging to a different category than
# the regex classifier predicts. Loaded here so they're easy to update.
EXCLUDE_IDS = [
    "B00I8C5ENU", "B01DA4V074", "B072BYCRNF", "B07PDSTG9C", "B07FSMS62P",
    "B089QJZ54T", "B07QYRQTYJ", "B07772JBX1", "B09XHS8485", "B00XZV0WA4",
    "B0043FOOVY", "B004UR9P9Q", "B01N4KVSPA", "B000OKW0IG", "B01K5XFK3I",
    "B00006AMS4", "B0000C9ZJX", "B004575BIU", "B095YFFLVB", "B07855HMW4",
    "B09LR9TLY3", "B074Y7XRB6", "B087YJ1NGC", "B07XCM94SN", "B074PYK939",
    "B01IC2W0IW", "B01B2YLS7G", "B076ZXWVVK", "B016YJWPSK",
]

# Manual overrides: parent_asin → correct category
MANUAL_CATEGORY_OVERRIDES = {
    # "other" → laptop
    "laptop": [
        "B08113LD2J", "B07FW86HB2", "B07MSBPJTK", "B01GAJ166Q", "B07Q7XKJG7",
        "B07XKBJX4R", "B07Y2DMY39", "B01MEESR3Q", "B097B3WVB3", "B01KKJT5P6",
    ],
    # "other" / phone → tablet
    "tablet": [
        "B07MZCJG4B", "B072NXBWVH", "B073W1P6MC", "B074JH8N5P", "B07JMJ1D54",
        "B06Y6KTD8G", "B076H9ZGFH", "B01L9NO7D2", "B01G6HDQTM", "B073X22PVP",
        "B01B3Z8WJG", "B06XNX91YB", "B00VQLJ4D6", "B073WXR1QB", "B07GVQK742",
        "B0849RN16P", "B07CY2SG9J", "B005890G84", "B06XB1TGYW", "B07G8FVW5W",
        "B01F7WBUQ8", "B09NQJG4CJ", "B08JB39D5N", "B07ZQR2Z69", "B07WJW3JX4",
        "B0753CL6CY", "B01ER4AX9M", "B01F1Z7DEE", "B01F5LWSS0", "B0721BQD9M",
        "140053271X", "B07XLNH4LM", "B07X7X5VTM", "B084WXRYMN", "B0BK31GHQW",
        "B07RD1R31Y", "B08428CB3R", "B017A6S0FS", "B00KDRQEYQ", "B005890FN0",
        "B089GVTVBD", "B08318VQBG", "B0872FTSYD", "B08SJ6YLYF", "B06ZZ6Z1YZ",
        "B07V5NB2VK", "B01NAI7MGR"
    ],
    # "other" → desktop
    "desktop": [
       "B08113LD2J", "B07FW86HB2", "B07MSBPJTK", "B01GAJ166Q", "B07Q7XKJG7",
        "B07XKBJX4R", "B07Y2DMY39", "B01MEESR3Q", "B097B3WVB3", "B01KKJT5P6",
        "B07NRWZYH9", "B072PR8GR4", "B07HDVHPSF", "B019ZTV1WM", "B01INVTPE4",
        "B07J9PQXPB", "B071NLF1BZ", "B079K7YGV9", "B07Z7H92T4", "B075FB3TYD",
        "B077BC21GJ", "B06VYFRF61", "B06X97G1S4", "B075RHSHLJ", "B07RFKL7GN",
        "B08NFKD18V", "B07H823NZM", "B099LLSBQW", "B00HWM5PJU", "B09BZW4SW4",
        "B008C7C1E6", "B073GDS46R", "B01M3Y3TWO", "B009JXJXAC", "B07MHXX3G2",
        "B08YZ9TG8R", "B07DM3VN9Q", "B01ER4AQ9O", "B01N5VKVB1", "B07D4ZMYCM",
        "B07V421MVB", "B0754YMJ67", "B073WQNF36", "B07MXQNYKZ", "B00TL7LNKY",
        "B092W9VPL8", "B00EC4QJAU", "B08WYLS34W", "B079T8JW58", "B00FMCBK6O",
        "B01KSJU8Y0", "B007CKOG42", "B083ZLFFPB", "B082P5TQ92", "B0793BSQQD",
        "B07K1N9752", "B01N5RBNI1", "B00BFFEBE0", "B09MR213L7", "B08777VPVX",
        "B01G2C5KJ0", "B079RLB975", "B0083PR78C", "B00E6J2SN8", "B00EY50MVO",
        "B00JXLGEC0", "B06W9P66FS", "B00CW7MKSY", "B009TM70IU", "B081YBRXZR",
        "B08ZRC54VH", "B095BX4SLL", "B08WYKDJ3G", "B07RDJ8PC5", "B009E9KKGW",
        "B00R45UMSE", "B00BBMGZF0", "B01D7EMWYC", "B01KZWCVQ8", "B0BFP9NRNL",
        "B0923DLGV4", "B00K67RPRS", "B01L3EXAC6", "B0094K2848", "B071Y5SZYR",
        "B076J8ZP9H", "B01C7UGSJC", "B01ER4AJWI", "B01HAQMR6G", "B00HZ4F3Z6",
        "B00AH4A950", "B0CD8XBXLM", "B08YLVBDM3", "B071KZ73LC", "B081Y3GFHZ",
        "B009AEPXAY", "B01DM29I7S", "B01BYG0VDY", "B00MVKUVLU", "B07MFDG8W1",
        "B076V56RD8", "B07CKB6D19", "B08Z9V7KWS", "B0BW1PVSQ9", "B088P9CQ6R",
        "B0098O6K3U", "B07MQ71YC3", "B09L1N5146", "B01M2X4G2K", "B017D79SL4",
        "B077GYQX6K", "B01HAQHE3M", "B073WQKF44", "B01M0O5MV6", "B01M7VW4M9",
        "B09VRRK8D9", "B0071N53EM", "B01MS01H6J", "B06ZZHMKPR", "B06ZXS286Y",
        "B072LSWFQD", "B07BYBK1CV", "B01E7Q07DW", "B01ER4ARCU", "B01F1Z787G",
        "B01MTF911G", "B07CR5RYDB", "B078TJ2Z7R", "B0BZJH8R4Q", "B07982CRJQ",
        "B07588SY5W", "B07VBY428R", "B07H7VQ166", "B08121F2C7", "B07YG7MPWS",
        "B07MTM9HTF", "B07HGVFBL1", "B083LBN5DD", "B0795ZRMYC", "B07CP1R5DV",
        "B00N4LFEI0", "B07CKLV8ZD", "B08741VZGL", "B07DFRKYSP", "B01M5BBIJS",
        "B077QNVH93", "B00CL8FW4S", "B005J2E2US", "B005J2DNHG", "B009JXK1T4",
        "B00AVYPLPY", "B01HAQHI7Y", "B07DS88P27", "B087NYZ8R7", "B087SFJ1HL",
        "B07BN2LR7Y", "B07F1XB5FK", "B077NVN4CW", "B01HAQBIVG", "B01BVX8BM8",
        "B07GN5Z96F", "B095B9F9XV", "B0BXH752KY", "B0838MYCKT", "B096CP7G6B",
        "B083M9V8M5", "B083M9ZWB6", "B084QD1QB7", "B00AW4OCZI", "B07CBFZ2Q3",
        "B07NL341NY", "B01I0560MS", "B0041RSE5Q", "B019FV6G34", "B083SS8MMZ",
        "B08SVF8GGN", "B08SVKJ6R6", "B08YMV3WT5", "B00N9CWWJS", "B08TBQYTDY",
        "B081565ZMY", "B072BGTQSY", "B07VYTJZFC", "B097WPNSGY", "B077J58D17",
        "B07G5QRMK4", "B01CU4E0KY", "B08Z2YCTC5", "B0C4CF7Y5X", "B082WHQJN7",
        "B071HYYB4M", "B07Y8W8F5Q", "B00Y9F4YUE", "B07PHM3934", "B076DHSSJF",
        "B01IG6NDNK", "B071CZMHZQ", "B077H2JY74", "B01B9HBP4C", "B01LD4ML30",
        "B00NC088EA", "B0141R3OV4", "B084QHRJLZ", "B072L7711F", "B079K1HBL5",
        "B073WQMKV4", "B07F2N9B59", "B09JL263CG", "B073WQHXR8", "B07PP1ZMQY",
        "B09C386RGG", "B00QE9UCZO", "B08HST79G4", "B00S1T2QBS", "B00CRJQG4G",
        "B07G66MSRH", "B0861G3BW1", "B01N2HO0VA", "B07W99R3F3", "B01K0OY2RM",
        "B08BZ3ZPG1", "B07BJXJ2R5", "B01C7UGPGS", "B077DN16ZX", "B01J42JPL4",
        "B019MJ3L78", "B004GCIYY2", "B004UR1670", "B008D9HZ62", "B01B24A3W2",
        "B01F1Z78RG", "B01F1Z78U8", "B06XSXFRV4", "B06XJFTYBL", "B07C4WKZMQ",
        "B07H82DZKS", "B075JSK7TR", "B07BFW9D9T", "B07SPFKBGV", "B07G36XGPM",
        "B07W5KSP3L", "B07F87JCMV", "B0BFCNKPCB", "B08BZ3XC8F", "B08BZ42SN1",
        "B09KQ3ZG2F", "B01M1BW7KK", "B07BFQWP7R", "B08288LGFQ", "B07D195XBL",
        "B07JVDVXJH", "B07HPYK24B", "B01MPXAPX0", "B074T13QP1", "B07RDJ8CGR",
        "B0C5HGQJBY", "B07LCGBGPF", "B0842V96QN", "B07SMX4QG2", "B009DXVVB2",
        "B00B4DIGXA", "B0B164K6GG", "B084L8Z4FP", "B084LJNVMC", "B07SC8X4GW",
        "B07RDJ8DRQ", "B01DO8F82Y", "B08121F9W2", "B08121HG8S", "B072STZXZ8",
        "B07JZ8DYPR", "B073ZND1FZ", "B07793T8LK", "B01LD4MGR6", "B06XDCRS12",
        "B00AAIC5S2", "B006ITMC7Q", "B006O7FLH0", "B014YN6YZ8", "B01N9C2C97",
        "B076CS5G1F", "B07F2KPLHD", "B0B3TSXVNF", "B07BZYQ4J2", "B01HAQG37A",
        "B07BX68HZY", "B087NXHDF8", "B08ZL6S7DS", "B07J5YBPZM", "B07DZJK8TK",
        "B079NNLCMG", "B075JLPLBL", "B07CMVB9J3", "B088JPPCFY", "B00TOW7T86",
        "B07RC92XDF", "B0B15NZRQT", "B09TL9DGK6", "B001R4L0Y8", "B098854S8C",
        "B07FP1QJ5P", "B071L1BPK8", "B077RCCYT8", "B07XKVHPRP", "B07ST48R6M",
        "B092BYWMNP", "B08CG9PZLR", "B09MHBY5HX", "B07HNJ2XKJ", "B008HBHHTG",
        "B0BWSD1XTX", "B0711WXD7Y", "B07SMND9DJ", "B08468B4FP", "B074MJRR36",
        "B079TGL2BZ", "B084TS138V", "B01N4FLSD0", "B004UR16ES", "B078NPYG9Y",
        "B077P34GJ2", "B081XL7H37", "B07MF4GQW6", "B07RXWPNG4", "B07NMSYLF4",
        "B0778HR29P", "B074HDXBC7", "B07PVCLSDZ", "B07DN3JNP6", "B07F35XY9N",
        "B07VTB2BF2", "B07HPWGP1X", "B08681C929", "B094W6QJY4", "B08MWQF4KG",
        "B07G2RG7JH", "B012B6YQF0", "B0956JKSH7", "B095C23PMJ", "B087QXGHSG",
        "B08KFP1Y1P", "B07SPQ37DS", "B083NR5R2G", "B082WMS2DT", "B08B17W9B6",
        "B07Y7P9P96", "B01M6CRZP4", "B08V5B4425", "B07H471WQT", "B001FWXFKE",
        "B01M5G7EQ3", "B0CFT86RRQ", "B09V8GB2CK", "B08J8GPPH2", "B08HXCGW87",
        "B09BMGCVZH", "B081Z793P8", "B09PFYW194", "B07GDDSJNH", "B0B6GJDM6W",
        "B07DN22J16", "B0C72SSP6X", "B09Y78B5GY", "B09QPP787G", "B00QX89YQ4",
        "B08HVB86ZK", "B01N1RM6N5", "B07XZP24L3", "B07MBSJYF9", "B07XQLJ7K3",
        "B07Y8QRX7W", "B0956JJVXG", "B07Z8GZH4P", "B08NWD6S2V", "B084MNHG2S",
        "B084P7T4NY", "B084QJRXFW", "B07HRJZY9D", "B07RKL5QL5", "B082PMG1GQ",
        "B071YPCBTS", "B075FLJ87G", "B0BHNFKVMV", "B0C6KL9LKQ", "B07G8Z3QQT",
        "B09W59DNH5", "B09JJ5MJX3", "B07QNM6LGW", "B09V6NMDGZ", "B07F21W1GX",
        "B09LJPW1SQ", "B01M7VVCSK", "B0C5S191L7", "B0C9QXRX79", "B087D8TH7W",
        "B07N75WXT2", "B09TDSWLCM", "B07XGNN51H", "B07SHNQ8FV", "B084RNT2JB",
        "B08HZ8WY6R", "B0B5KR26ZL", "B08TCXDZWS", "B09NJWJL85", "B0886GXT6L",
        "B07797QV9M", "B08JHHK1G1", "B08SVGYD5D", "B08SW98K2G", "B08SVF5RFG",
        "B07S5MNMT2", "B07BBDLRMB", "B081D7C6CD", "B08558L133", "B08121T49K",
        "B005E18T94", "B08V9DVJGK", "B089S9N86P", "B08HMWYRK6", "B07YXSLFDH",
        "B0867HF7XR", "B07RFKK878", "B07VVMF51P", "B089RWZZ1G", "B07F73YGZJ",
        "B081217DYJ", "B0811ZZMP9", "B08L8W2WJ8", "B092CMW6K9", "B083NQ3KP8",
        "B083JNK965", "B086L5P6D1", "B08XWDZ22S", "B071HZRD36", "B07SGZG3ZV",
        "B08NTCH21H", "B09KQY4RWR", "B07BJYD4QT", "B09DB2MG5M", "B084GQ1HG4",
        "B08ZBZ651Q", "B09XWMPBM5", "B088FY251L", "B07YNTR349", "B002QK22YO",
        "B08CBLWPQS", "B09QK7MT7T", "B07PDHTGQ2", "B0C3G9GZH1", "B0CFKFW9MM",
        "B0BW8PGDSV", "B083ZMDKZG", "B08R6L1Z2L", "B08K99DGJF", "B083MVFXP8",
        "B081FJNWD4", "B07Y25H9Z3", "B07XB3B53Q", "B081226VGQ", "B07RDJ8HLC",
        "B0787HV7WM", "B071XTV7JC", "B087D7XTNT", "B099MBMQYL", "B07QC8L7K4",
        "B088FHF29H", "B07DN3FPKK", "B098TS26YZ", "B08WTQGJ55", "B0CGRDQ17V",
        "B07BK2XLHV", "B083M9BWXM", "B085F2VJCK", "B0787HK4MW", "B08ZKY469Y",
        "B09HLBM2CL", "B07QRZJRK9", "B08BK3SDT9", "B08L9WM9SF", "B08H8F1954",
        "B07HGQ7R1M", "B096T8KH67", "B08TCYDJ79", "B09B4KYWBG", "B08K3MY5K5",
        "B08K3LW7ZT", "B08SHL527P", "B07HZS73R1", "B087DV319X", "B0C1KQQXD6",
        "B08FGJ67ZL", "B08FGJB9X4", "B096L3KTLQ", "B007C5VOP6", "B07892Z7YL",
        "B071P6QWMH", "B072LMGSRW", "B01N59TYCO", "B086Z2L8M8", "B09F6MLXSZ",
        "B018VOO1LU", "B01B7QHJMW", "B008PB7VPI", "B07HFJX4G2", "B08VD39G88",
        "B07TMRF1GD", "B06XTVHWYQ", "B06X6GFQHG", "B07JKHN5WP", "B0CCSDV35W",
        "B08WCKF6WN", "B096QWS9T9", "B00CE5DIFS", "B09T3WHJDL", "B095DQ5WCJ",
        "B0993X4HP2", "B07JHFWMBH", "B01COKKSVY", "B08CCJ5QF5", "B081212H4B",
        "B093BJ7Z4R", "B076JRHXJG", "B0816ZWM5X", "B07VJG269Z", "B08N5RQBG8",
        "B09RQPX26Z", "B09TGXCPZC", "B079Y8R8PV", "B06XHKRZYX", "B09YGSVMVK",
        "B07LFDTSP5", "B09576KJTB", "B08ZSH1MYX", "B07C347PSD", "B0C3XW248Z",
        "B07R81HFCY", "B07VBH9TWS", "B07SWR3NVJ", "B07BT44PHY"
    ],
}

# Products confirmed as "other" (accessories, cables, etc.) — drop them
REAL_OTHER_IDS = [
    "B07HM9CX87", "B079MGHY92", "B00H713QY2", "B07HP9X5T7", "B0B3D5V422",
    "B088JWLKDN", "B00I8C5ENU", "B074Y8Q2Z8", "B003FWHHYC", "B07XWXX2CB",
]

# Regex patterns used by the auto-classifier
_PHONE_RE = re.compile(
    r"\b(iphone|galaxy[\s\-]?[sz]\d|pixel[\s\-]?\d|oneplus[\s\-]?\d"
    r"|smartphone|android\s+phone)\b", re.I
)
_TABLET_RE = re.compile(
    r"\b(ipad|kindle|fire\s+tablet|galaxy\s+tab|tab[\s\-]?[ae]\d"
    r"|lenovo\s+tab|surface\s+go)\b", re.I
)
_LAPTOP_RE = re.compile(
    r"\b(laptop|notebook|macbook|chromebook|surface\s+pro|thinkpad"
    r"|ideapad|inspiron|pavilion|spectre|envy|zenbook|vivobook)\b", re.I
)
_DESKTOP_RE = re.compile(
    r"\b(desktop|imac|mac\s+mini|mac\s+pro|mac\s+studio|all[\s\-]in[\s\-]one"
    r"|tower\s+pc|mini\s+pc|nuc)\b", re.I
)


def _classify_title(title: str) -> str:
    t = str(title).lower()
    if _DESKTOP_RE.search(t):
        return "desktop"
    if _LAPTOP_RE.search(t):
        return "laptop"
    if _TABLET_RE.search(t):
        return "tablet"
    if _PHONE_RE.search(t):
        return "phone"
    return "other"


def classify_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Auto-classify from product title
    df["category"] = df["product_title"].apply(_classify_title)

    # Apply manual overrides (asin → correct category)
    for category, asins in MANUAL_CATEGORY_OVERRIDES.items():
        mask = df["parent_asin"].isin(asins)
        df.loc[mask, "category"] = category

    # Drop confirmed "other" products (accessories, etc.)
    df = df[~df["parent_asin"].isin(REAL_OTHER_IDS)]

    # Drop everything still tagged "other" or "phone"
    df = df[~df["category"].isin(["other", "phone"])]

    return df



#  Section 5: Brand extraction


_BRAND_RE = re.compile(
    r"\b(apple|iphone|ipad|macbook|imac"
    r"|samsung|galaxy"
    r"|dell|alienware|xps"
    r"|hp|hewlett[\s\-]?packard|omen|spectre|pavilion|envy|elitebook|probook"
    r"|lenovo|thinkpad|ideapad|yoga"
    r"|asus|zenbook|vivobook|rog|tuf"
    r"|acer|aspire|nitro|predator|swift"
    r"|microsoft|surface"
    r"|amazon|kindle|fire|echo"
    r"|google|pixel|chromebook"
    r"|lg|sony|toshiba|razer)\b",
    re.I,
)


def _detect_brand(title: str) -> str:
    m = _BRAND_RE.search(str(title).lower())
    if not m:
        return "unknown"
    raw = m.group(0).lower().replace("-", "").replace(" ", "")
    # Normalise aliases
    aliases = {
        "hewlettpackard": "hp",
        "iphone": "apple", "ipad": "apple",
        "macbook": "apple", "imac": "apple",
        "galaxy": "samsung",
        "alienware": "dell", "xps": "dell",
        "thinkpad": "lenovo", "ideapad": "lenovo", "yoga": "lenovo",
        "zenbook": "asus", "vivobook": "asus", "rog": "asus", "tuf": "asus",
        "aspire": "acer", "nitro": "acer", "predator": "acer", "swift": "acer",
        "kindle": "amazon", "fire": "amazon", "echo": "amazon",
        "pixel": "google", "chromebook": "google",
        "omen": "hp", "spectre": "hp", "pavilion": "hp",
        "envy": "hp", "elitebook": "hp", "probook": "hp",
        "surface": "microsoft",
    }
    return aliases.get(raw, raw)


def extract_brands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["brand"] = df["product_title"].apply(_detect_brand)
    return df


#  Section 6: Language detection + translation

def translate_non_english(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and translate non-English reviews (best-effort)."""
    try:
        from langdetect import detect, DetectorFactory
        from deep_translator import GoogleTranslator

        DetectorFactory.seed = 0

        def _safe_detect(text: str) -> str:
            try:
                return detect(str(text))
            except Exception:
                return "en"

        df = df.copy()
        df["language"] = df["review_text"].apply(_safe_detect)
        non_en = ~df["language"].isin(["en", "unknown"])
        n = non_en.sum()
        print(f"  Translating {n:,} non-English reviews …")

        for lang, group in df[non_en].groupby("language"):
            indices = group.index
            translated = []
            for text in group["review_text"]:
                try:
                    translated.append(
                        GoogleTranslator(source=lang, target="en").translate(str(text))
                    )
                except Exception:
                    translated.append(text)
            df.loc[indices, "review_text"] = translated

        print("  Translation complete.")
    except ImportError:
        print("langdetect / deep-translator not installed — skipping translation.")
        df["language"] = "en"

    return df

#  Section 7: Vader sentiment

def add_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    vader = SentimentIntensityAnalyzer()
    df = df.copy()
    df["vader_sentiment"] = df["review_text"].apply(
        lambda t: vader.polarity_scores(str(t))["compound"]
    )
    return df



#  Section 8: Tradeoff flag

CONTRAST_PATTERN = r"\b(but|however|although|though|despite|unfortunately|except|yet|still|while|whereas)\b"

def add_tradeoff_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_tradeoff"] = df["review_text"].astype(str).str.contains(
        CONTRAST_PATTERN, case=False, regex=True
    ).astype(int)
    return df


#  Section 9: Save results

def save_results(df: pd.DataFrame) -> None:
    if USE_LOCAL_CSV:
        os.makedirs("dataset", exist_ok=True)
        print(f"Saving {len(df):,} rows to {LOCAL_OUTPUT_CSV} …")
        df_out = df.copy()
        df_out["date"] = df_out["date"].astype(str)
        df_out.to_csv(LOCAL_OUTPUT_CSV, index=False)
        print(f"Saved. File size: {os.path.getsize(LOCAL_OUTPUT_CSV) / 1e6:.1f} MB")
    else:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID)
        print(f"  Uploading {len(df):,} rows to {OUTPUT_TABLE} …")
        df_out = df.copy()
        df_out["date"] = df_out["date"].astype(str)
        job = client.load_table_from_dataframe(
            df_out,
            OUTPUT_TABLE,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        )
        job.result()
        print("Upload complete.")


#  Entry point

def run() -> None:
    mode = "LOCAL CSV" if USE_LOCAL_CSV else "BIGQUERY"
    
    print(f"STEP 3 — EDA Data Preparation  [{mode}]")


    df = load_reviews()
    df = basic_clean(df)
    df = assign_price_tiers(df)
    df = assign_covid_period(df)
    df = classify_categories(df)
    df = extract_brands(df)
    df = translate_non_english(df)
    df = add_vader_sentiment(df)
    df = add_tradeoff_flag(df)

    save_results(df)
    print("Step 3 complete.\n")


if __name__ == "__main__":
    run()
