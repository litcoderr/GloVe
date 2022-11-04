import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from glove.app import GloVe


MSRVTT_ROOT = Path("/data/MSRVTT")
TEXT = MSRVTT_ROOT / "annotation" / "MSR_VTT.json"
FEAT_ROOT = MSRVTT_ROOT / "feat" / "glove" / "text"
FEAT_ROOT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    app = GloVe(
        vectors_file="/GloVe/pretrained/glove.840B.300d.txt"
    )

    with open(TEXT, "r") as f:
        json_dict = json.load(f)
    annotation_data = json_dict["annotations"]

    for data in tqdm(annotation_data):
        txt_id = data["id"]
        txt = data["caption"]
        txt_vec = app.get_vector_seq(txt)

        vec_path = FEAT_ROOT / f"{str(txt_id).zfill(8)}.npy"
        with open(vec_path, "wb") as f:
            np.save(f, txt_vec)
