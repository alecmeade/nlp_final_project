import os
import argparse
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from speech2image.modelbiggan import Speech2Image
from dataloaders.image_caption_dataset import ImageCaptionDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=1, help="How many GPUs to train with.")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_speech2image", help="Model location.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--dataset", type=str, default="./data/PlacesAudioEnglish", help="Dataset path.")
    parser.add_argument("--niter", type=int, default=100, help="Number of training iters.")
    parser.add_argument("--version", type=str, default=None, help="Experiment version.")
    args = parser.parse_args()

    # Model dir
    folder = args.checkpoints_dir
    if not os.path.isdir(folder):
        os.makedirs(folder)

    cuda = torch.cuda.is_available()
    train_set = ImageCaptionDataset(os.path.join(args.dataset, "samples.json"))
    train_dl = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=8, pin_memory=cuda, batch_size=args.batch_size)

    # Main model
    model = Speech2Image()
    
    # Model training
    logger = loggers.WandbLogger(args.version, args.checkpoints_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoints_dir, args.version),
        filename="image2reverb_{epoch:04d}",
        period=10,
        save_top_k=-1,
        verbose=True,
    )
    
    trainer = Trainer(
        gpus=args.n_gpus if cuda else None,
        auto_select_gpus=True,
        accelerator="ddp" if cuda else None,
        max_epochs=args.niter,
        default_root_dir=args.checkpoints_dir,
        callbacks=[checkpoint_callback],
        logger=logger
    )
    
    trainer.fit(model, train_dl)


if __name__ == "__main__":
    main()
