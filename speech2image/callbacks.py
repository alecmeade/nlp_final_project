import os
import subprocess
import shutil
import pytorch_lightning as pl
import torchvision


class ImageWriterCallback(pl.callbacks.Callback):
    def __init__(self, dirpath, combine_img_and_audio=True):
        self.dir = dirpath
        self.combine = combine_img_and_audio
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for i in range(len(outputs["I_AUDs"])):
            print("Writing %d." % i)
            a_name = os.path.basename(outputs["I_AUDs"][i])
            example_path = os.path.join(self.dir, os.path.splitext(a_name)[0])
            if not os.path.isdir(example_path):
                os.makedirs(example_path)
                
            f_img = os.path.join(example_path, "f.png")
            r_img = os.path.join(example_path, "r.png")
            i_aud = os.path.join(example_path, a_name)
            
            torchvision.utils.save_image(outputs["G_IMGs"][i], f_img)
            torchvision.utils.save_image(outputs["R_IMGs"][i], r_img)
            shutil.copy2(outputs["I_AUDs"][i], i_aud)

            if self.combine:
                subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", f_img, "-i", i_aud, "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", os.path.join(example_path, "example.mp4")])
