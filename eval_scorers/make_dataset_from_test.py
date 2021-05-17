import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default=None, help="Path to the output.")
    parser.add_argument("--src_set", type=str, default=None, help="Source dataset to use as lookup.")
    parser.add_argument("--outfile", type=str, default="data/test_output.json", help="Path to the output.")
    args = parser.parse_args()

    d = {k + "_base_path":"" for k in ["audio", "image"]}
    d["data"] = []
    
    with open(args.src_set) as json_file:
        audio_keys = {os.path.basename(x["wav"]):x for x in json.load(json_file)["data"]}

    for subdir in os.listdir(args.test_dir):
        s = os.path.join(args.test_dir, subdir)
        print("Processing %s." % s)
        if os.path.isdir(s):
            files = os.listdir(s)
            apath = [f for f in files if os.path.splitext(f)[1] == ".wav"][0]
            info = audio_keys[apath]
            info["category"] = "/" + info["image"][:info["image"].rfind("/")]
            d["data"].append({**audio_keys[apath], **{
                "wav": os.path.join(s, apath),
                "image": os.path.join(s, "f.png")
            }})
        
        with open(args.outfile, "w") as json_file:
            json.dump(d, json_file, indent=4)
            



if __name__ == "__main__":
    main()
