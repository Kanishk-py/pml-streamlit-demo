import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


# Load the MNIST dataset
def load_dataset():
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
	mnist_train = datasets.MNIST('./data', train=True, transform=transform, download=True)
	mnist_test = datasets.MNIST('./data', train=False, transform=transform, download=True)
	train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
	test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
	random_test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True)

	return train_loader, test_loader, random_test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, random_test_loader = load_dataset()
random_sampler = iter(random_test_loader)


# Define the CNN architecture
class CNN(nn.Module):
	def __init__(self, dropout_rate=0.3, N=5):
		super(CNN, self).__init__()
		self.N = N  # Number of samples
		self.filters = 4
		self.conv1 = nn.Conv2d(1, self.filters, kernel_size=3)
		self.fc1 = nn.Linear(self.filters * 13 * 13, 16)
		self.fc2 = nn.Linear(16, 10)
		self.dropout = nn.Dropout(p=dropout_rate)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2)
		x = x.view(-1, self.filters * 13 * 13)
		x = self.dropout(x)
		x = F.relu(self.fc1(x))

		x_tmp = self.dropout(x)  # MC Dropout
		x_list = self.fc2(x_tmp).unsqueeze(0)
		for i in range(self.N-1):
			x_tmp = self.dropout(x)
			x_tmp = self.fc2(x_tmp).unsqueeze(0)
			x_list = torch.cat([x_list, x_tmp], dim=0)

		return x_list

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def accuracy(model, test_loader):
	model.eval()
	model.apply(apply_dropout)
	with torch.no_grad():
		correct = 0
		total = 0
		print("Calculating Accuracy")
		for images, labels in test_loader:
			images, labels = images.to(device), labels.to(device)
			output = torch.softmax(model(images), dim=2).data
			_, predicted = torch.max(output.mean(dim=0), dim=1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

		print(f"Test Accuracy: {(correct / total) * 100:.2f}%")

# Train the model
def train_save_model(model, train_loader, device, epochs=10):
	model = CNN().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss().cuda()

	for epoch in range(epochs):
		model.train()
		for images, labels in train_loader:
			images, labels = images.to(device), labels.to(device)
			# images, labels = Variable(images), Variable(labels)

			optimizer.zero_grad()
			logits = model(images).mean(0)

			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

		print(f"Epoch {epoch+1}/{epochs} | Training Loss: {loss.item():.4f}")
		accuracy(model, test_loader)

		# Save the model
		torch.save(model.state_dict(), f'models/mnist_cnn_model_{epoch}.pth')

train_save_model(CNN(), train_loader, device, epochs=15)