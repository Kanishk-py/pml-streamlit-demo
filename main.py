import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Load the MNIST dataset
@st.cache_data
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

def expected_calibration_error(model, test_loader, title=None, M=10, device='cpu'):
	# uniform binning approach with M number of bins
	bin_boundaries = np.linspace(0, 1, M + 1)
	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]
	test_loader_iter = iter(test_loader)

	data, target = next(test_loader_iter)
	true_labels = target.to(device)
	output = torch.softmax(model(data.to(device)), dim=2).data
	confidences, predicted_labels = torch.max(output, 2)

	# predicted_labels = []
	# true_labels = []
	# confidences = []
	reliabilities = []

   # keep confidences / predicted "probabilities" as they are
	model.eval()
	model.apply(apply_dropout)

	with torch.no_grad():
		for data, target in test_loader_iter:
			target = target.to(device)
			output = model(data.to(device))
			output = torch.softmax(output, dim=2).data
			conf, predicted = torch.max(output, 2)
			predicted_labels = torch.cat([predicted_labels, predicted], dim=1)
			true_labels = torch.cat([true_labels, target], dim=0)
			confidences = torch.cat([confidences, conf], dim=1)

	true_labels = true_labels.unsqueeze(1).expand(-1, predicted_labels.shape[0]).T.cpu().numpy()
	predicted_labels = predicted_labels.cpu().numpy()
	confidences = confidences.cpu().numpy()


	# get a boolean list of correct/false predictions
	accuracies = predicted_labels==true_labels
	eces = np.zeros(predicted_labels.shape[0])

	for n_sample in range(predicted_labels.shape[0]):
		temp_rel = []
		for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
			# determine if sample is in bin m (between bin lower & upper)
			in_bin = np.logical_and(confidences[n_sample] > bin_lower.item(), confidences[n_sample] <= bin_upper.item())
			# can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
			prop_in_bin = in_bin.astype(float).mean()

			if prop_in_bin.item() > 0:
				# get the accuracy of bin m: acc(Bm)
				accuracy_in_bin = accuracies[n_sample][in_bin].astype(float).mean()
				# get the average confidence of bin m: conf(Bm)
				avg_confidence_in_bin = confidences[n_sample][in_bin].mean()
				# calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
				eces[n_sample] += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
				bin_accuracy = np.mean(true_labels[n_sample][in_bin]==predicted_labels[n_sample][in_bin])
				temp_rel.append(bin_accuracy)
			else:
				temp_rel.append(0.0)  # Avoid division by zero
		reliabilities.append(temp_rel)

	if title:
		# Plot the reliability diagram
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal reference line
		for rel, ece in zip(reliabilities, eces):
			ax.plot(np.linspace(0, 1, M), rel, marker="o", linestyle="--", label=f"ECE = {ece:.4f}")
		ax.legend(loc="upper left")
		ax.set_xlabel("Mean Predicted Probability")
		ax.set_ylabel("Empirical Accuracy")
		ax.set_title(f"Reliability Diagram (ECE = {eces.mean():.4f}, (Std = {eces.std():.4f}))")
		ax.grid(True)
		# ax.savefig(title)
		return eces, fig
	return eces

# Evaluate the model accuracy
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

		return (correct / total) * 100

# Streamlit UI
st.title("MC Dropout for Uncertainty Estimation on MNIST")

# Sliders for dropout rate and number of samples
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 1.0, 0.5, 0.05)
num_samples = st.sidebar.slider("Number of Samples", 1, 20, 5, 1)
epoch_trained = st.sidebar.slider("Epoch Trained", 1, 15, 10, 1)

if st.sidebar.button("Run MC Dropout"):
	st.write("Evaluating with MC Dropout...")

	model = CNN(dropout_rate=dropout_rate, N=num_samples).to(device)
	model.load_state_dict(torch.load(f'models/mnist_cnn_model_{epoch_trained-1}.pth', map_location=torch.device(device)))
	model.eval()

	# acc, uncertainty = accuracy(model, test_loader)

	ece, fig = expected_calibration_error(model, test_loader, title="Reliability Diagram", M=10, device=device)
	st.pyplot(fig)


if st.sidebar.button("Select a Random Sample"):
	st.write("Selecting a random sample...")

	model = CNN(dropout_rate=dropout_rate, N=num_samples).to(device)
	model.load_state_dict(torch.load(f'models/mnist_cnn_model_{epoch_trained-1}.pth', map_location=torch.device(device)))
	model.eval()

	st.write("Accuracy: ", accuracy(model, test_loader))

	data, target = next(random_sampler)
	output = torch.softmax(model(data.to(device)), dim=2).data
	confidences, predicted_labels = torch.max(output, 2)
	# print(confidences, predicted_labels)

	fig, ax = plt.subplots(figsize=(6, 6))
	ax.imshow(data.squeeze().numpy(), cmap='gray')
	ax.set_title(f"True Label: {target.item()}, Predicted Labels: {predicted_labels.cpu().numpy().flatten()}")
	ax.set_xticks([])
	ax.set_yticks([])
	st.pyplot(fig)

if st.sidebar.button("Plot ECE"):
	model = CNN(dropout_rate=dropout_rate, N=num_samples).to(device)
	ece = []
	for i in range(15):
		model.load_state_dict(torch.load(f'models/mnist_cnn_model_{i}.pth', map_location=torch.device(device)))
		model.eval()

		ece.append(expected_calibration_error(model, test_loader, M=10, device=device))

	ece = np.array(ece)
	fig, ax = plt.subplots(figsize=(6, 6))
	for i in range(num_samples):
		ax.plot(np.arange(1, 16), ece[:, i], marker="o", linestyle="--", alpha=0.6)
	ax.plot(np.arange(1, 16), ece.mean(axis=1), marker="o", linestyle="-", color='k', label='Mean ECE')
	ax.set_title(f"ECE v/s Epoch")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("ECE")
	ax.set_xticks(np.arange(1,16))
	ax.legend()
	# plt.show(fig)
	st.pyplot(fig)

	ece = []
	for dropout_rate in np.linspace(0, 0.9, 10):
		model = CNN(dropout_rate=dropout_rate, N=5).to(device)
		model.load_state_dict(torch.load(f'models/mnist_cnn_model_14.pth', map_location=torch.device(device)))
		model.eval()

		ece.append(expected_calibration_error(model, test_loader, M=10, device=device))

	ece = np.array(ece)

	fig, ax = plt.subplots(figsize=(6, 6))
	for i in range(5):
		ax.plot(np.linspace(0, 0.9, 10), ece[:, i], marker="o", linestyle="--", alpha=0.4)
	ax.plot(np.linspace(0, 0.9, 10), ece.mean(axis=1), marker="o", linestyle="-", color='k', label='Mean ECE')
	ax.set_title(f"ECE v/s Dropout Rate")
	ax.set_xlabel("Dropout Rate")
	ax.set_ylabel("ECE")
	ax.set_xticks(np.linspace(0, 0.9, 10))
	ax.legend()
	# plt.show(fig)
	st.pyplot(fig)
