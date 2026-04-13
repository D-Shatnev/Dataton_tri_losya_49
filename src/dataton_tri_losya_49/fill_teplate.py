import csv
import tqdm

from dataton_tri_losya_49.components.embedder.base_line import ONNXEmbedder
from dataton_tri_losya_49.components.embedder.gmm import GMMUBMEmbedder
from dataton_tri_losya_49.components.embedder.speech_brain import SpeechBrainEmbedder

from dataton_tri_losya_49.vector_db import search_similar_vectors

def fill_template(template_path, resolver):
    files_to_index = {}

    embedder=None
    match resolver:
        case "baseline":
            embedder = ONNXEmbedder("./secrets/baseline.onnx", use_mono_mean=True)
        case "gmm":
            embedder = GMMUBMEmbedder("./ubm_64.pkl")
        case "speechbrain":
            embedder = SpeechBrainEmbedder()
    
    with open(template_path) as f:
        reader = csv.reader(f)
        next(reader)
        for idx, line in tqdm.tqdm(enumerate(reader)):
            if idx > 1250:
                break
            files_to_index[f"{line[0]}"] = idx

    with open(f"{resolver}_test.csv", "w") as f:
        f.write("file,neighbours\n")
        for file in tqdm.tqdm(files_to_index):
            query = embedder.extract_embedding(file)
            n = search_similar_vectors(query, resolver, 10)
            ids = [v["id"] for v in n]
            f.write(f"{file},\"{','.join([str(i) for i in ids])}\"\n")

fill_template("./train_template.csv", "speechbrain")