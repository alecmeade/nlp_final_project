import os
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infiles", nargs="+", help="Path to two+ datasets.")
    parser.add_argument("--outfile", type=str, default=None, help="Path to output dataset.")
    parser.add_argument("--outdir", type=str, default="./data", help="Path to output directory.")
    args = parser.parse_args()
    
    if not args.outfile:
        args.outfile = "_".join([os.path.splitext(os.path.basename(x))[0] for x in args.infiles]) + ".json"
        
    if os.path.basename(args.outdir) not in args.outfile:
        args.outfile = os.path.join(args.outdir, args.outfile)
        print("Setting outfile to %s." % args.outfile)
    
    d = {"data": {}}
    for f in args.infiles:
        with open(f) as json_file:
            df = json.load(json_file)
            print("Loaded %s: %d examples." % (f, len(df["data"])))
            for k, v in df.items():
                if k == "data":
                    d["data"].update({x["uttid"]:x for x in v})
                else:
                    d.setdefault(k, v)
    d["data"] = list(d["data"].values())
    
    with open(args.outfile, "w") as json_file:
        json.dump(d, json_file, indent=4)
    
    print("Wrote %d examples to file %s." % (len(d["data"]), args.outfile))


if __name__ == "__main__":
    main()
