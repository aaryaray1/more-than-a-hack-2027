import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from validation.idsse import SOURCE_DIR  # noqa: E402

MIRROR = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
MATCHES = {"J03WMX": "000001", "J03WN1": "000001", "J03WOH": "000002", "J03WOY": "000002",
           "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002"}
KINDS = ("DFL_02_01_matchinformation", "DFL_04_03_positions_raw_observed")


def main():
    os.makedirs(SOURCE_DIR, exist_ok=True)
    for match_id, competition in sorted(MATCHES.items()):
        for kind in KINDS:
            name = "{}_DFL-COM-{}_DFL-MAT-{}.xml".format(kind, competition, match_id)
            path = os.path.join(SOURCE_DIR, name)
            if os.path.isfile(path):
                print("have   {}".format(name))
                continue
            print("fetch  {}".format(name))
            urllib.request.urlretrieve(MIRROR + name, path + ".part")
            os.replace(path + ".part", path)


if __name__ == "__main__":
    main()
