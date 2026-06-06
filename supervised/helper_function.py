from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import json

# walk through each of the subdirectories and count the file present
def walk_through_dir(dir_path = "."):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """ 
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# Check image afer transform
def plot_transformed_images(image_paths, transform,n=3, seed:int= None):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    if seed:
        torch.manual_seed(seed)

    for i in range(n):
        random_idx = torch.randint(0, len(image_paths), size=(1,)).item()
        img_path = image_paths[random_idx]
        img = Image.open(img_path)
        transformed_img = transform(img)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
             
        ax[1].imshow(transformed_img.permute(1, 2, 0))
        ax[1].set_title("Transformed Image")  
        ax[1].axis("off")

        fig.suptitle(f"Class: {img_path.parent.stem}")

  


def find_class(dir_path: Path):
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        dir_path (Path): Target directory path.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. get the class name by scanning the target directory
    classes = sorted(item.name for item in os.scandir(dir_path) if item.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        print(f"Couldn't find any class folder in {dir_path}.")
    
    # 3. Create a class to index mapping
    class_to_idx = {idx: cls_name for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx



class CustomImageDataset(Dataset):
    """
    Dataset

    Args:
        dir_path (Path): Target directory path.
        transform: transformation

    """
    # Initialize with a target directory and transforms
    def __init__(self, dir_path: Path, transform=None) -> None:
        # get all image paths
        self.paths = list(pathlib.Path(dir_path).glob("*/*.jpg")) # update this if your image format is not .jpg
        # setup transforms
        self.transform = transform
        self.classes, self.class_to_idx = find_class(dir_path)
    
    # Make function to load images
    def load_image(self, index: int) -> Image.Image:
        # Open the image at index
        img_path = self.paths[index]
        return Image.open(img_path)
    
    #Overwrite the _len__ method to return the length of the dataset
    def __len__(self) -> int:
        "Returns the total number of samples in the dataset."
        return len(self.paths)
    
    # Overwrite the __getitem__ method to return a sample from the dataset at the given index
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X,y)."
        img = self.load_image(index)
        class_name = self.paths[index].parent.stem
        class_idx = next((k for k,v in self.class_to_idx.items() if v == class_name), None)

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return the image and the label
        else:
            return img, class_idx # return the image and the label
        


def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: list[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    # 3. Set random seed
    if seed:
        torch.manual_seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = torch.randint(0, len(dataset), (n,))

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)


# train loop

def train_step(model:torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches

    for batch, (X,y) in enumerate(dataloader):
        # send data to target device
        X, y = X.to(device), y.to(device)

        # 1. forward pass
        y_pred =  model(X)

        # 2. Calculate and accummulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
    


# test step

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


from tqdm.auto import tqdm
from torch import nn

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
def save_model(model_name:str, model_result:dict, model: torch.nn.Module):
    MODEL_PATH = Path("/.models")
    MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                    exist_ok=True # if models directory already exists, don't error
    )

    # Create model save path
    MODEL_NAME = model_name + ".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
            f=MODEL_SAVE_PATH)
    
    with open(f"./models/{model_name}_result.json", "w") as j:
        json.dump(model_result, j, indent=4)


# # Spearate train data to train and validation
import pathlib
from torchvision.io import read_image
from torchvision.utils import save_image

def dir_seperate(train_dir):
    flower_classes = os.listdir(train_dir)
    print(f"flower classes: {flower_classes}")

    for flower_class in flower_classes:
        img_paths = list(pathlib.Path(train_dir).glob(f"{flower_class}/*.jpg"))
        torch.manual_seed(42)
        for i in range(int(len(img_paths)*0.2)):
            random_img_path = img_paths[torch.randint(len(img_paths), size=(1,)).item()]
            img= read_image(str(random_img_path))/255.0
            path_validation = os.path.join(".","data","validation", random_img_path.parent.stem, random_img_path.name)
            path_train = os.path.join(".","data","train", random_img_path.parent.stem, random_img_path.name)
            save_image(img, path_validation, normalize=True)
            os.remove(str(path_train))
            img_paths = list(pathlib.Path(train_dir).glob("*/*.jpg"))
