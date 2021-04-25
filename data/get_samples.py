import os
import json
import shutil

json_file = "/data/sls/d/corpora/PlacesAudioEnglish/metadata/train.json"

dest_path = "/usr/users/iapalm/nlp_final_project/data/PlacesAudioEnglish"

with open(json_file, "r") as f:
    json_data = json.load(f)

image_base_path = json_data["image_base_path"]
audio_base_path = json_data["audio_base_path"]
json_sample_data = json_data["data"]

json_export = dict()
json_export["image_base_path"] = "data/PlacesAudioEnglish/images/"
json_export["audio_base_path"] = "data/PlacesAudioEnglish/"
json_export["data"] = list()

n_samples = 10
for i in range(n_samples):
    sample = json_sample_data[i]

    wav_file = sample["wav"]
    image_file = sample["image"]

    wav_dir = os.path.dirname(wav_file)
    image_dir = os.path.dirname(image_file)

    wav_dest_dir = os.path.join(dest_path, wav_dir)
    image_dest_dir = os.path.join(os.path.join(dest_path, "images"), image_dir)

    print(f"Making dir: {wav_dest_dir}")
    print(f"Making dir: {image_dest_dir}")
    os.makedirs(wav_dest_dir, exist_ok=True)
    os.makedirs(image_dest_dir, exist_ok=True)

    wav_original = os.path.join(audio_base_path, wav_file)
    image_original = os.path.join(image_base_path, image_file)

    print(f"Copy file: {wav_original} to {wav_dest_dir}")
    print(f"Copy file: {image_original} to {image_dest_dir}")
    shutil.copy(wav_original, wav_dest_dir)
    shutil.copy(image_original, image_dest_dir)

    json_export["data"].append(sample)

with open("samples.json", "w+") as f:
    json.dump(json_export, f)

print("done")
