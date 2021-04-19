transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.30691805, 122.95039414, 113.86538318), (1, 1, 1)),
    transforms.RandomCrop(32, padding=4)
]) 
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.30691805, 122.95039414, 113.86538318), (1, 1, 1)),
    transforms.RandomCrop(32, padding=4)  
])