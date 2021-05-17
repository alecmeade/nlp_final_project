import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from dataloaders.image_caption_dataset import ImageCaptionDataset
from .classifier_scorer import ClassifierScorer
from .davenet_scorer import DaveNetScorer

CLASSES_205_PATH = "googlenet_places205/categoryIndex_places205.csv"
NIKHIL_CAFFE_GOOGLENET_PLACES205_PATH = "googlenet_places205/snapshot_iter_765280.caffemodel.pt"
CAFFE_GOOGLENET_PLACES205_PATH = "googlenet_places205/2755ce3d87254759a25cd82e3ca86c4a.npy"
GOOGLENET_PLACES205_PATH = "googlenet_places205/googlenet_places205.pth"
DAVENET_PATH = "davenet_vgg16_MISA_1024_pretrained/"

MODEL_TYPES = {
    "googlenet": GOOGLENET_PLACES205_PATH,
    "googlenetcaffe": CAFFE_GOOGLENET_PLACES205_PATH,
    "vgg16caffe": NIKHIL_CAFFE_GOOGLENET_PLACES205_PATH
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes205", type=str, default=CLASSES_205_PATH, help="Path to category indices.")
    parser.add_argument("--model_dir", type=str, default="./eval_scorers/trained_models", help="Path to pretrained models.")
    parser.add_argument("--model_type", type=str, default="googlenetcaffe", help="Path to pretrained models.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model.")
    parser.add_argument("--dataset", type=str, default="data/test_seen_2020.json", help="Path to the dataset.")
    parser.add_argument("--davenet_path", type=str, default=DAVENET_PATH, help="Path to DaveNet dir.")
    parser.add_argument("--audio_model", type=str, default=None, help="Path to the DaveNet audio model.")
    parser.add_argument("--image_model", type=str, default=None, help="Path to the DaveNet image model.")
    parser.add_argument("--outdir", type=str, default="./scores", help="Output dir for plots, reports, etc.")
    parser.add_argument("--store_logits", action="store_true", help="Store classifier output logits.")
    parser.add_argument("--device", type=str, default="cuda", help="Either cuda or cpu.")
    args = parser.parse_args()
    
    # Check args
    args.classes205 = os.path.join(args.model_dir, args.classes205)

    if not args.model_path:
        args.model_path = os.path.join(args.model_dir, MODEL_TYPES[args.model_type])
    
    if not os.path.isdir(args.davenet_path):
        args.davenet_path = os.path.join(args.model_dir, args.davenet_path)
    
    if not args.audio_model:
        args.audio_model = os.path.join(args.davenet_path, "audio_model.pth")
    
    if not args.image_model:
        args.image_model = os.path.join(args.davenet_path, "image_model.pth")
    
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # Load scorers
    dave_scorer = DaveNetScorer(args.audio_model, args.image_model)
    clf_scorer = ClassifierScorer(args.model_path, args.model_type)
    
    # Dataset
    audio_conf = {"use_raw_length": True}
    dataset = ImageCaptionDataset(args.dataset, audio_conf=audio_conf, normalize=True)
    data_key = {x["wav"]:x for x in dataset.data}
    loader = torch.utils.data.DataLoader(dataset, num_workers=8, pin_memory=args.device == "cuda", batch_size=1)

    # Other
    RGB_mean = [0.485, 0.456, 0.406]
    RGB_std = [0.229, 0.224, 0.225]
    lnet_img_transform = transforms.Compose([transforms.Resize(224), transforms.Normalize(mean=RGB_mean, std=RGB_std)])

    c205_c2n = {}
    c205_n2c = {}
    with open(args.classes205) as infile:
        for l in infile.readlines():
            c, n = l.split(" ")
            n = int(n)
            c205_c2n[c] = n
            c205_n2c[n] = c

    def get_idx(l):
        img_path = l["image"]
        try:
            return c205_c2n["/" + img_path[:img_path.rfind("/")]]
        except:
            return c205_c2n[l["category"]]
    
    def get_classname(idx):
        return c205_n2c[idx]
    
    # Score dataset
    scores = {}
    for img, audio, _, apath in tqdm(loader):
        heatmap, matches, sisa, misa, sima = dave_scorer.score(audio.squeeze(0), img.squeeze(0))
        lnet_img = lnet_img_transform(img)
        l = data_key[apath[0][apath[0].find("wavs"):] if "wavs" in apath[0] else apath[0]]
        uid = l["uttid"]
        i = ""
        while (uid + str(i)) in scores:
            i = i + 1 if i != "" else 1
        uid = uid + str(i)
        scores[uid] = {}
        scores[uid]["y"] = get_idx(l)
        logits = clf_scorer.score(lnet_img)
        scores[uid]["y_pred"] = logits.argmax(dim=1).item()
        scores[uid]["logits"] = logits.squeeze().detach().cpu().numpy().tolist()
        scores[uid]["c"] = get_classname(scores[uid]["y"])
        scores[uid]["c_pred"] = get_classname(scores[uid]["y_pred"])
        scores[uid]["sisa"] = sisa
        scores[uid]["misa"] = misa
        scores[uid]["sima"] = sima
        
        with open(os.path.join(args.outdir, "results.json"), "w") as json_file:
            json.dump(scores, json_file, indent=4)


if __name__ == "__main__":
    main()
