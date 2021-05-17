import os
import argparse
import torch
from pytorch_lightning import Trainer
from speech2image.callbacks import ImageWriterCallback
from speech2image.model import Speech2Image
from dataloaders.image_caption_dataset import ImageCaptionDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=1, help="How many GPUs to train with.")
    parser.add_argument("--output_dir", type=str, default="./output_speech2image", help="Output location.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--dataset", type=str, default="./data/flickr_audio/samples.json", help="Dataset path.")
    parser.add_argument("--model", type=str, default=None, help="Path to a full model to test with.")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of iterations over the set.")
    parser.add_argument("--version", type=str, default=None, help="Experiment version.")
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    test_set = ImageCaptionDataset(args.dataset, audio_conf={"audio_type": "audio"})
    indices = torch.arange(len(test_set.data)).repeat(args.n_trials)
    test_set = torch.utils.data.Subset(test_set, indices)
    test_dl = torch.utils.data.DataLoader(test_set, shuffle=False, num_workers=8, pin_memory=cuda, batch_size=args.batch_size)

    # Main model
    model = Speech2Image()
    if args.model:
        m = torch.load(args.model, map_location=model.device)
        model.load_state_dict(m["state_dict"], strict=False)
    
    # Model training
    dirpath = os.path.join(args.output_dir, args.version)
    output_callback = ImageWriterCallback(dirpath=dirpath, combine_img_and_audio=False)
    
    trainer = Trainer(
        gpus=args.n_gpus if cuda else None,
        auto_select_gpus=True,
        accelerator="ddp" if cuda else None,
        callbacks=[output_callback]
    )
    
    trainer.test(model, test_dl)


if __name__ == "__main__":
    main()
