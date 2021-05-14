import os
import argparse
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from speech2image.model import Speech2Image
from dataloaders.image_caption_dataset import ImageCaptionDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=1, help="How many GPUs to train with.")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_speech2image", help="Model location.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--dataset", type=str, default="./data/flickr_audio/samples.json", help="Dataset path.")
    parser.add_argument("--val", type=str, default=None, help="Dataset path.")
    parser.add_argument("--niter", type=int, default=100, help="Number of training iters.")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to a pretrained GAN model.")
    parser.add_argument("--model", type=str, default=None, help="Path to a full model to resume training.")
    parser.add_argument("--version", type=str, default=None, help="Experiment version.")
    args = parser.parse_args()

    # Model dir
    folder = args.checkpoints_dir
    if not os.path.isdir(folder):
        os.makedirs(folder)

    cuda = torch.cuda.is_available()
    train_set = ImageCaptionDataset(args.dataset, image_conf={"center_crop": True}, audio_conf={"audio_type": "audio"})
    train_dl = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=8, pin_memory=cuda, batch_size=args.batch_size)
    val_set = ImageCaptionDataset(args.val, image_conf={"center_crop": True}, audio_conf={"audio_type": "audio"})
    val_dl = torch.utils.data.DataLoader(val_set, shuffle=True, num_workers=8, pin_memory=cuda, batch_size=args.batch_size)

    # Main model
    model = Speech2Image(pretrained=args.pretrained)
    if args.model:
        m = torch.load(args.model, map_location=model.device)
        model.load_state_dict(m["state_dict"])
    
    # Model training
    logger = loggers.WandbLogger(args.version, args.checkpoints_dir)

    dirpath = os.path.join(args.checkpoints_dir, args.version)
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="speech2image_{epoch:04d}",
        # period=10,
        every_n_val_epochs=1,
        save_top_k=-1,
        verbose=True
        # save_last=True
    )
    
    trainer = Trainer(
        gpus=args.n_gpus if cuda else None,
        auto_select_gpus=True,
        limit_val_batches=4,
        accelerator="ddp" if cuda else None,
        max_epochs=args.niter,
        default_root_dir=args.checkpoints_dir,
        callbacks=[checkpoint_callback],
        logger=logger
    )
    
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
