from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed
import glob

class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]

class MVTecAT(Dataset):
    def __init__(self, root_dir, defect_name, size, transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            defect_name (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size
        
        # Find test images
        if self.mode == "train":
            # Training mode: match all images under train/good (supports mixed-case suffixes)
            train_good_dir = self.root_dir / defect_name / "train" / "good"
            # Explicitly match multiple suffix variants (png/PNG, jpg/JPG, jpeg/JPEG, etc.)
            self.image_names = [
                Path(p) for p in glob.glob(str(train_good_dir / "*.png")) +
                               
                                glob.glob(str(train_good_dir / "*.jpg")) +

                                glob.glob(str(train_good_dir / "*.jpeg")) +
                                glob.glob(str(train_good_dir / "*.JPEG"))
            ]
            # self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.{png,JPG,jpeg}"))
            print("loading images")
            # During training we cache the resized images for performance reasons (not a good coding style)
            # self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            self.imgs = Parallel(n_jobs=10)(
                delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file)
                for file in self.image_names
            )
            print(f"loaded {len(self.imgs)} images")
        else:
            # Test mode
            # self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.{png,JPG,jpeg}")))
            # Testing mode: match images in all subfolders under test (supports mixed-case suffixes)
            test_dir = self.root_dir / defect_name / "test"
            # Match images under any subdirectory ("*") with multiple suffix variants
            self.image_names = [
                Path(p) for p in glob.glob(str(test_dir / "*/*.png")) +
                          
                                glob.glob(str(test_dir / "*/*.jpg")) +

                                glob.glob(str(test_dir / "*/*.jpeg")) +
                                glob.glob(str(test_dir / "*/*.JPEG"))
            ]
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img

        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size,self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "good"
