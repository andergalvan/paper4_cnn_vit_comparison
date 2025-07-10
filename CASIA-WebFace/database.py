from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_dataloaders(image_size, train_path, validation_path, close_test_path, batch_size, num_workers, pin_memory):
    
    """ Load training, validation and CSR testing dataloaders. """
    
    # Transformation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # Datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    validation_dataset = datasets.ImageFolder(root=validation_path, transform=transform)
    close_test_dataset = datasets.ImageFolder(root=close_test_path, transform=transform)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    close_test_loader = DataLoader(close_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    
    return train_loader, validation_loader, close_test_loader


def load_openset_dataloader(image_size, openset_path, batch_size, num_workers, pin_memory):
    
    """Load the dataloader for the OSR scenario testing."""
    
    # Transformation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Load dataset and dataloader
    opentest_dataset = datasets.ImageFolder(root=openset_path, transform=transform)
    opentest_loader = DataLoader(opentest_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    return opentest_loader
