import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchinfo import summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

from skimage.metrics import structural_similarity as ski_ssim
from pytorch_msssim import SSIM, MS_SSIM

def main():
    # Setup device to GPU if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = 'cards' # Selected dataset contained in "data/" directory

    train_dir = Path(f"data/{dataset_name}/train")
    test_dir = Path(f"data/{dataset_name}/test")

    # Parameters
    NUM_WORKERS = os.cpu_count()
    BATCH_SIZE = 32
    IMG_RESIZE = 256     # Resize of an input images
    INPUT_CROP  = 256    # Crop of resized images (when equal to IMG_RESIZE value, there is no crop applied)
    TRAINING_SIZE = 1000 # Number of images taken for training
    TEST_SIZE = 100      # Number of images taken for testing

    def explore_dir(dir_path):
        """
        Function for exploring specified directory dependencies and number of images
        
        Args:
            dir_path (str or Path): Location of interested directory   
        """
        for current_dir, dirs, img in os.walk(dir_path):
            print(f"'{current_dir}' have {len(dirs)} directories and {len(img)} images.")
        print("")

    explore_dir(train_dir)
    explore_dir(test_dir)

    images_paths = list(test_dir.glob("*/*.jpg")) # Gets list of all images paths from given directory
    random_image_path = random.choice(images_paths)
    img = Image.open(random_image_path)

    print(f"Random image path: {random_image_path}")
    print(f"Image size: {img.height}x{img.width}") 

    plt.imshow(img)
    plt.axis("off")

    ## DATA AUGMENTATION
    data_transform = transforms.Compose([
        transforms.Resize(size=(IMG_RESIZE, IMG_RESIZE)),
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()])

    def image_transformation_plot(image_path, transform):
        """
        Plots and compares selected image before and after transformation.
        
        Args:
            image_path (Path): Path of image to plot
            transform (Pytorch Transforms): Selected transforms composition
        """
        f = Image.open(image_path)
        
        fig, ax = plt.subplots(1, 2)
        plt.gray()
        ax[0].imshow(f) 
        ax[0].set_title(f"Original \nSize: {f.size}")
        ax[0].axis("off")
        
        transformed_image = transform(f).permute(1, 2, 0) 
        ax[1].imshow(transformed_image) 
        ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
        ax[1].axis("off")

    image_transformation_plot(random_image_path, data_transform)

    def random_crop(data_dir, N, crop):
        """
        Function for building dataset with randomly choosen images with applied random crop to  size.
        
        Args:
            data_dir (str or Path): Location of interested directory
            N (int): Number of images in prepared dataset
            
        Returns:
            output_dataset (list): A list of cropped images
        """
        crop_size = (crop, crop)

        output_dataset = []

        images_paths = list(data_dir.glob("good/*.jpg"))
        print(images_paths)

        for i in range(0, N):
            random_image_path = random.choice(images_paths)
            img = Image.open(random_image_path)
            img = data_transform(img).squeeze(0)
            
            x_start = random.randint(0, img.shape[0]-crop_size[0])
            y_start = random.randint(0, img.shape[1]-crop_size[1])
            x_end = x_start + crop_size[0]
            y_end = y_start + crop_size[1]
            
            crop = img[x_start:x_end, y_start:y_end]
            crop = crop.type(torch.float32).unsqueeze(0)
            output_dataset.append(crop)

        return output_dataset

    train_data = random_crop(data_dir=train_dir, N=TRAINING_SIZE, crop = INPUT_CROP)

    test_data = random_crop(data_dir=test_dir, N=TEST_SIZE, crop = INPUT_CROP)

    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=BATCH_SIZE, 
                                num_workers=NUM_WORKERS, 
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=BATCH_SIZE, 
                                num_workers=NUM_WORKERS, 
                                shuffle=False)

    # Check if the input values are in <0,1> range to define activation function or normalize
    images = next(iter(train_dataloader))
    print(f"Image shape: {images.shape}\n {torch.min(images)}\n {torch.max(images)}")

    class autoencoder_v0(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            
            self.encoder = nn.Sequential( 
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d( 32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d( 64, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d( 128, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d( 64, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d( 128, 500, 3, stride=1, padding=0)
            )

            self.decoder = nn.Sequential(

                nn.ConvTranspose2d(500, 128, 3, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d( 64, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d( 128, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d( 64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d( 32, 1, 4, stride=2, padding=1),
                nn.Sigmoid()
            )
            
        
        def forward(self, x: torch.Tensor):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        
    model = autoencoder_v0()

    summary(model, input_size=[BATCH_SIZE, 1, INPUT_CROP, INPUT_CROP])

    def train_step(model, dataloader, loss_fn, optimizer):
        
        model.train()
        train_loss = 0
            
        for batch, img in enumerate(dataloader):
            
            img = img.to(device)
            recon = model(img)
            loss = 1 - loss_fn(recon, img)
            train_loss += loss.item() 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss = train_loss / len(dataloader)
        
        return train_loss

    def test_step(model, dataloader, loss_fn, epoch):
        
        model.eval()
        test_loss = 0
        
        with torch.inference_mode():
            for batch, img in enumerate(dataloader):

                img = img.to(device)
                recon = model(img)
                loss = 1 - loss_fn(recon, img)
                test_loss += loss.item() 
        
            test_loss = test_loss / len(dataloader)
        
        return test_loss
    def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs):
        """
        Running train and test steps, plotting training progress and loss values.
        
        Args:
            model: Autoencoder model instance
            train_dataloader: Dataloader with training images
            test_dataloader: Dataloader with test images
            optimizer: Used optimizer - SSIM or MSE
            loss_fn: Used loss function
            epochs: Number of epochs to train the model
            
        Returns:
            results (dict): A dictionary with calculated loss values
        """
        results = {"train_loss": [], "test_loss": []}
        
        for epoch in tqdm(range(epochs)):
            train_loss = train_step(model=model,
                                    dataloader=train_dataloader,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer)
            
            test_loss = test_step(model=model,
                                dataloader=test_dataloader,
                                loss_fn=loss_fn,
                                epoch=epoch)
            
            train_ssim = (1 - train_loss)*100
            test_ssim = (1 - test_loss)*100
            
    
            print(f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"train_ssim: {train_ssim:.2f}% | "
                f"test_ssim: {test_ssim:.2f}% | ")

            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_loss)

        return results

    criterion = SSIM(win_sigma=1.5, data_range=1, size_average=True, channel=1)
    # criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-4)
    NUM_EPOCHS = 200

    # model_results = train(model=model, 
    #                     train_dataloader=train_dataloader,
    #                     test_dataloader=test_dataloader,
    #                     optimizer=optimizer,
    #                     loss_fn=criterion, 
    #                     epochs=NUM_EPOCHS)

    # ax = pd.DataFrame({
    #     'Train Loss': [loss for loss in model_results['train_loss']],
    #     'Test Loss': [loss for loss in model_results['test_loss']]
    # }).plot(title='SSIM Loss Decrease', logy=True)

    # ax.set_xlabel("Epochs")
    # ax.set_ylabel("Loss")

    def load_model(model_name):
        """
        Function for loading saved models parameters. Neural network architecture should be the same as on saved model.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            loaded_model: A model instance with loaded parameters
        """
        model_path = Path("models")
        model_save_path = model_path / model_name

        loaded_model = autoencoder_v0()
        loaded_model.load_state_dict(torch.load(f=model_save_path))

        loaded_model = loaded_model.to(device)
        
        return loaded_model
        


    # model_path = Path("models")
    # model_name = "cards.5acc.pth"
    # model_save_path = model_path / model_name
    # model_path.mkdir(parents=True, exist_ok=True)

    # print(f"Saving model to: {model_save_path}")

    # torch.save(obj=model.state_dict(), f=model_save_path)

    model_name = "cards.5acc.pth"
    model_loaded = load_model(model_name)

    data_transform_test = transforms.Compose([
        transforms.Resize(size=(INPUT_CROP, INPUT_CROP)),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])

    def image_reconstructed_plot(model: torch.nn.Module, 
                                image_path: str,  
                                transform=None,
                                device: torch.device = device):
        
        """ Plots image before and after passing through autoencoder, and SSIM contour """
        
        fig, ax = plt.subplots(1, 3)
        plt.gray()   
        
        img = Image.open(image_path)
        
        if transform:
            img_transformed = transform(img)

        model.to(device)
        model.eval()
        with torch.inference_mode():
            input_image = img_transformed.unsqueeze(dim=0).to(device)
            reconstructed = model(input_image)
        
        original = img_transformed.permute(1, 2, 0).cpu()
        
        defect = image_path.split("/")
        
        ax[0].imshow(original) 
        ax[0].set_title(f"{defect[3].capitalize()}")
        ax[0].axis("off")
        
        reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).cpu()
        ax[1].imshow(reconstructed) 
        ax[1].set_title("Reconstructed")
        ax[1].axis("off")
        
        img_old = np.array(original.squeeze(2))
        img_new = np.array(reconstructed.squeeze(2))
        _, S = ski_ssim(img_old, img_new, full=True, channel_axis=False, data_range=1.0)

        ax[2].imshow(1-S, vmax = 1, cmap='jet') 
        ax[2].set_title("SSIM")
        ax[2].axis("off")

        plt.axis(False)
        plt.show()  # Add this line to display the reconstructed image


    train_image_path = ("data/cards/test/good/002.jpg")
    test_image_path_1 = ("data/cards/test/different_name/001.png")

    image_reconstructed_plot(model=model_loaded,
                            image_path=train_image_path,
                            transform=data_transform,
                            device=device)

    image_reconstructed_plot(model=model_loaded,
                            image_path=test_image_path_1,
                            transform=data_transform,
                            device=device)


if __name__ == "__main__":
    main()
