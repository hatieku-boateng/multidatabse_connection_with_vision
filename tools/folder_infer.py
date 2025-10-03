from torchvision import datasets, transforms

root = r"D:\2025\MSC\python programming\advance python\Project folder 1\id_cards\curated"
tfm = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
])

ds = datasets.ImageFolder(root=root, transform=tfm)

print("Classes:", ds.classes)          # ['drivers_licence', 'ghana_card', 'voter_id']
print("Mapping:", ds.class_to_idx)     # {'drivers_licence':0, 'ghana_card':1, 'voter_id':2}
print("First sample:", ds[0][0].shape, ds[0][1])  # (C,H,W), label index
