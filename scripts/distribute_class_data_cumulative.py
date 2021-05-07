import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default=None, help="Path to a dataset.")
    parser.add_argument("--phase", type=str, default=None, help="One of train|test|dev (inferred from filename if None).")
    parser.add_argument("--type", type=str, default=None, help="One of seen|unseen (inferred from filename if None).")
    parser.add_argument("--pathsrc", type=str, default=None, help="Source of image and audio paths; will default to infile if None.")
    parser.add_argument("--outdir", type=str, default="data/dataset_", help="Out directory (should be just a prefix, for example data/places1cl_).")
    args = parser.parse_args()
    
    # Set flags as necessary
    if not args.phase:
        args.phase = check_which(args.infile, ["train", "test", "dev"], throw=True)
        print("Setting phase to %s." % args.phase)
        
    if not args.type:
        args.type = check_which(args.infile, ["seen", "unseen"])
        if args.type != "train":
            print("Setting type to %s." % args.type)
    
    if not args.pathsrc:
        args.pathsrc = args.infile
    
    args.outdir = args.outdir + args.phase + args.type
    
    # Open dataset
    with open(args.infile) as json_file:
        d = json.load(json_file)
    
    with open(args.pathsrc) as json_file:
        paths = json.load(json_file)
        paths["data"] = []
    
    # Get classnames
    outd = {}
    for x in d["data"]:
        cn = get_classname(x)
        outd.setdefault(cn, paths.copy())
        outd[cn]["data"].append(x)
    
    # Write files
    for k, v in outd.items():
        fn = os.path.join(args.outdir, "%s.json" % k)
        print("Writing %s." % fn)
        with open(fn, "w") as json_file:
            json.dump(v, json_file, indent=4)


def get_classname(x):
    return x["image"].split("/")[1]


def check_which(x, t, throw=True):
    j = []
    for term in t:
        if term in x:
            j.append(term)
    if len(j):
        return max(j, key=len)
    if throw:
        raise Exception("None of the provided terms matched.")
    else:
        return ""


if __name__ == "__main__":
    main()
