import numpy as np
import torch
from functools import partial
import copy
from torchvision import datasets, transforms
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.decomposition import PCA


def train_epoch(model, device, train_loader, optimizer, epoch, criterion= torch.nn.functional.cross_entropy):
	model.train()  # Setting the model to train mode

	loss_history = []
	accuracy_history = []

	for batch_idx, (data, target) in enumerate(train_loader):
		data= data.to(device)
		target= target.to(device)
		output = model.forward(data)
		probas= model.compute_proba(output)

		loss = criterion(probas, target)

		loss.backward()

		optimizer.step()

		optimizer.zero_grad()

		pred = output.argmax(dim=1, keepdim=True)
		correct = pred.eq(target.view_as(pred)).sum().item()

		loss_history.append(loss.item())
		accuracy_history.append(correct / len(data))

		if batch_idx % max(1, (len(train_loader.dataset) // len(data) // 10)) == 0:
			print(
				f"Train Epoch: {epoch}-{batch_idx} batch_loss={loss.item()/len(data):0.2e} batch_acc={correct/len(data):0.3f}"
			)

	return loss_history, accuracy_history


@torch.no_grad()
def validate(model, device, val_loader, criterion):
	model.eval()  # Setting model to evaluation mode
	test_loss = 0
	correct = 0
	for data, target in val_loader:
		data, target = data.to(device), target.to(device)
		output = model(data)
		test_loss += criterion(output, target).item() * len(data)
		pred = output.argmax(
			dim=1, keepdim=True
		)  # get the index of the max log-probability
		correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(val_loader.dataset)

	print(
		"Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
			test_loss,
			correct,
			len(val_loader.dataset),
			100.0 * correct / len(val_loader.dataset),
		)
	)
	return test_loss, correct / len(val_loader.dataset)


@torch.no_grad()
def get_predictions(model, device, val_loader, criterion, num=None):
	model.eval()
	points = []
	for data, target in val_loader:
		data, target = data.to(device), target.to(device)
		output = model(data)
		loss = criterion(output, target)
		pred = output.argmax(dim=1, keepdim=True)

		data = np.split(data.cpu().numpy(), len(data))
		loss = np.split(loss.cpu().numpy(), len(data))
		pred = np.split(pred.cpu().numpy(), len(data))
		target = np.split(target.cpu().numpy(), len(data))
		points.extend(zip(data, loss, pred, target))

		if num is not None and len(points) > num:
			break
	return points


def run_mnist_training(model, num_epochs, lr, batch_size, device, optimization_algo, seed=42):
	# ===== Data Loading =====
	# Input images are normalized
	transform = transforms.Compose(
		[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
	)
	train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)

	# Using the test set as a validation set
	val_set = datasets.MNIST("./data", train=False, transform=transform)

	g = torch.Generator()
	g.manual_seed(seed)

	train_loader = torch.utils.data.DataLoader(
		train_set,
		batch_size=batch_size,
		shuffle=True,  # Can be important for training
		pin_memory=torch.cuda.is_available(),
		drop_last=True,
		num_workers=2,
		generator=g
	)
	val_loader = torch.utils.data.DataLoader(
		val_set,
		batch_size=batch_size,
	)

	# Model, Optimizer and Criterion
	optimizer = optimization_algo(model.parameters(), lr=lr)
	criterion = torch.nn.functional.cross_entropy

	# Learning rate scheduler (decrease lr_new= factor*old_lr if no improvement in accuracy for 3 epoch in a row)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', patience= 3, factor= 0.5, verbose= True)


	# Train Model
	new_w= get_weights(model)
	M= np.zeros(shape= (len(new_w), num_epochs + 1))
	M[:, 0]= new_w.clone()
	train_loss_history = []
	train_acc_history = []
	val_loss_history = []
	val_acc_history = []
	for epoch in range(1, num_epochs + 1):
		train_loss, train_acc = train_epoch(
			model, device, train_loader, optimizer, epoch, criterion
		)
		train_loss_history.extend(train_loss)
		train_acc_history.extend(train_acc)

		with torch.no_grad():
			w= get_weights(model)
			M[:, epoch]= w.clone()

		val_loss, val_acc = validate(model, device, val_loader, criterion)
		val_loss_history.append(val_loss)
		val_acc_history.append(val_acc)

		prev_lr = optimizer.param_groups[0]['lr']
		scheduler.step(val_loss)
		new_lr = optimizer.param_groups[0]['lr']

		if new_lr != prev_lr:
			print(f"⚠️ Learning rate changed at epoch {epoch}: {prev_lr:.2e} → {new_lr:.2e}")

	# ===== Plot training curves =====
	n_train = len(train_acc_history)
	t_train = num_epochs * np.arange(n_train) / n_train
	t_val = np.arange(1, num_epochs + 1)
	plt.figure()
	plt.plot(t_train, train_acc_history, label="Train")
	plt.plot(t_val, val_acc_history, label="Val")
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")

	plt.figure()
	plt.plot(t_train, train_loss_history, label="Train")
	plt.plot(t_val, val_loss_history, label="Val")
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Loss")

	# ===== Plot low/high loss predictions on validation set =====
	points = get_predictions(
		model,
		device,
		val_loader,
		partial(torch.nn.functional.cross_entropy, reduction="none"),
	)
	points.sort(key=lambda x: x[1])
	plt.figure(figsize=(15, 6))
	for k in range(5):
		plt.subplot(2, 5, k + 1)
		plt.imshow(points[k][0].reshape(28, 28), cmap="gray")
		plt.title(f"true={int(points[k][3])} pred={int(points[k][2])}")
		plt.subplot(2, 5, 5 + k + 1)
		plt.imshow(points[-k - 1][0].reshape(28, 28), cmap="gray")
		plt.title(f"true={int(points[-k-1][3])} pred={int(points[-k-1][2])}")

	return M


def retrain_on_new_w(model, num_epochs, lr, batch_size, device, optimization_algo, new_w, old_w, dir1, dir2, alpha, beta):
	# ===== Data Loading =====
	# Input images are normalized
	transform = transforms.Compose(
		[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
	)
	train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)

	# Using the test set as a validation set
	val_set = datasets.MNIST("./data", train=False, transform=transform)

	train_loader = torch.utils.data.DataLoader(
		train_set,
		batch_size=batch_size,
		shuffle=True,  # Can be important for training
		pin_memory=torch.cuda.is_available(),
		drop_last=True,
		num_workers=2,
	)
	val_loader = torch.utils.data.DataLoader(
		val_set,
		batch_size=batch_size,
	)

	# Model, Optimizer and Criterion
	optimizer = optimization_algo(model.parameters(), lr=lr)
	criterion = torch.nn.functional.cross_entropy

	# Learning rate scheduler (decrease lr_new= factor*old_lr if no improvement in validation loss for 3 epoch in a row)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', patience= 3, factor= 0.5)


	# Train Model
	alphas_history= [alpha]
	betas_history= [beta]
	set_weights(model, new_w)
	optimizer.zero_grad()

	M= np.zeros(shape= (len(new_w), num_epochs + 1))
	M[:, 0]= new_w.clone()
	train_loss_history = []
	train_acc_history = []
	val_loss_history = []
	val_acc_history = []
	for epoch in range(1, num_epochs + 1):
		train_loss, train_acc = train_epoch(
			model, device, train_loader, optimizer, epoch, criterion
		)
		train_loss_history.extend(train_loss)
		train_acc_history.extend(train_acc)

		with torch.no_grad():
			w= get_weights(model)
			M[:, epoch]= w.clone()
			delta_w= w-old_w
			alpha= torch.dot(delta_w, dir1)
			beta= torch.dot(delta_w, dir2)
			alphas_history.append(alpha.item())
			betas_history.append(beta.item())

		val_loss, val_acc = validate(model, device, val_loader, criterion)
		val_loss_history.append(val_loss)
		val_acc_history.append(val_acc)

		prev_lr = optimizer.param_groups[0]['lr']
		scheduler.step(val_loss)
		new_lr = optimizer.param_groups[0]['lr']

		if new_lr != prev_lr:
			print(f"⚠️ Learning rate changed at epoch {epoch}: {prev_lr:.2e} → {new_lr:.2e}")


	# ===== Plot training curves =====
	n_train = len(train_acc_history)
	t_train = num_epochs * np.arange(n_train) / n_train
	t_val = np.arange(1, num_epochs + 1)
	plt.figure()
	plt.plot(t_train, train_acc_history, label="Train")
	plt.plot(t_val, val_acc_history, label="Val")
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")

	plt.figure()
	plt.plot(t_train, train_loss_history, label="Train")
	plt.plot(t_val, val_loss_history, label="Val")
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Loss")

	plt.figure()
	plt.plot(alphas_history, betas_history, marker='o')
	plt.quiver(alphas_history[:-1], betas_history[:-1], 
			[a2 - a1 for a1, a2 in zip(alphas_history[:-1], alphas_history[1:])],
			[b2 - b1 for b1, b2 in zip(betas_history[:-1], betas_history[1:])],
			scale_units='xy', angles='xy', scale=1, color='blue')
	plt.xlabel("Alpha")
	plt.ylabel("Beta")
	plt.title("Adam trajectory in Alpha-Beta space")
	plt.axis('equal')

	# ===== Plot low/high loss predictions on validation set =====
	points = get_predictions(
		model,
		device,
		val_loader,
		partial(torch.nn.functional.cross_entropy, reduction="none"),
	)
	points.sort(key=lambda x: x[1])
	plt.figure(figsize=(15, 6))
	for k in range(5):
		plt.subplot(2, 5, k + 1)
		plt.imshow(points[k][0].reshape(28, 28), cmap="gray")
		plt.title(f"true={int(points[k][3])} pred={int(points[k][2])}")
		plt.subplot(2, 5, 5 + k + 1)
		plt.imshow(points[-k - 1][0].reshape(28, 28), cmap="gray")
		plt.title(f"true={int(points[-k-1][3])} pred={int(points[-k-1][2])}")

	return M


def get_weights(model):
    return torch.cat([p.data.view(-1).clone() for p in model.parameters()])

def set_weights(model, flattened_new_weights):
    idx = 0  # Start index in the flat weight tensor
    for p in model.parameters():
        shape = p.data.size()         # Get the shape of the current parameter
        numel = p.data.numel()        # Get the total number of elements in the parameter
        # Extract the corresponding values from flat_weights, reshape, and copy
        p.data.copy_(flattened_new_weights[idx:idx+numel].view(shape))
        idx += numel  # Move to the next segment in the flat weight tensor

def get_filterwise_normalized_direction(model):
    direction_parts = []

    for param in model.parameters():
        if param.ndim == 2:  # Only normalize weight matrices (e.g., Linear layers)
            # Each row is a filter (a neuron)
            rand = torch.randn_like(param)
            direction = torch.zeros_like(param)

            for i in range(param.size(0)):
                d_row = rand[i]
                θ_row = param.data[i]

                d_norm = d_row.norm()
                θ_norm = θ_row.norm()

                if d_norm > 0:
                    direction[i] = d_row / d_norm * θ_norm
                else:
                    direction[i] = torch.zeros_like(d_row)  # Safe fallback
        else:
            # For biases or other 1D tensors: scale entire vector
            rand = torch.randn_like(param)
            d_norm = rand.norm()
            θ_norm = param.data.norm()

            if d_norm > 0:
                direction = rand / d_norm * θ_norm
            else:
                direction = torch.zeros_like(rand)

        direction_parts.append(direction.view(-1))

    # Flatten and concatenate all directions
    return torch.cat(direction_parts)

def analyse_landscape(model, X, y, dir1, dir2, x_grid, y_grid, criterion= torch.nn.functional.cross_entropy):
	model.eval()

	current_weights= get_weights(model)
	
	loss_surface= np.zeros((len(x_grid), len(y_grid)))
	for i, alpha in enumerate(x_grid):
		for j, beta in enumerate(y_grid):
			with torch.no_grad():
				w = current_weights + alpha * dir1 + beta * dir2
				set_weights(model, w)
				output = model(X)
				probas= model.compute_proba(output)
				loss = criterion(probas, y)
			loss_surface[i, j] = loss.item()

	set_weights(model, current_weights)
	return loss_surface

def evaluate_chunk(grid_chunk, model, current_weights, dir1, dir2, X, y, criterion):
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    results = []

    for i, j, alpha, beta in grid_chunk:
        w = current_weights + alpha * dir1 + beta * dir2
        set_weights(model_copy, w)
        with torch.no_grad():
            output = model_copy(X)
            probas = model_copy.compute_proba(output)
            loss = criterion(probas, y)
        results.append((i, j, loss.item()))
    
    return results

def split_grid(x_grid, y_grid, n_chunks):
    grid = [(i, j, alpha, beta) 
            for i, alpha in enumerate(x_grid) 
            for j, beta in enumerate(y_grid)]
    chunk_size = len(grid) // n_chunks
    return [grid[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks-1)] + [grid[(n_chunks-1)*chunk_size:]]

def analyse_landscape_parallel(model, X, y, dir1, dir2, x_grid, y_grid, criterion= torch.nn.functional.cross_entropy, n_jobs= -1):
	model.eval()

	current_weights= get_weights(model)
	
	n_chunks = joblib.cpu_count() if n_jobs == -1 else n_jobs
	print(f"Launching computation for {n_chunks} chunks on {joblib.cpu_count()} processes")
	grid_chunks = split_grid(x_grid, y_grid, n_chunks)

	results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_chunk)(
            chunk, model, current_weights, dir1, dir2, X, y, criterion
        ) for chunk in grid_chunks
    )

	loss_surface = np.zeros((len(x_grid), len(y_grid)))
	for chunk in results:
		for i, j, loss_val in chunk:
			loss_surface[i, j] = loss_val

	set_weights(model, current_weights)
	return loss_surface

@torch.no_grad()
def compute_traj_pca(model, X, y, M, num_epochs, n_points= 30):
	M_new= np.zeros_like(M)
	for i in range(num_epochs):
		M_new[:, i]= M[:, i] - M[:, -1]

	pca= PCA(n_components=2)
	_= pca.fit_transform(M_new.T)
	pca_1= pca.components_[0]
	pca_2= pca.components_[1]

	explained_var = pca.explained_variance_ratio_

	alphas_history= []
	betas_history= []
	for i in range(num_epochs+1):
		alpha= torch.dot(torch.tensor(M_new[:, i]), torch.tensor(pca_1))
		beta= torch.dot(torch.tensor(M_new[:, i]), torch.tensor(pca_2))
		alphas_history.append(alpha.item())
		betas_history.append(beta.item())
	
	x_grid= np.linspace(- (max(alphas_history) - min(alphas_history)) - 1, (max(alphas_history) - min(alphas_history)) + 1, n_points)
	y_grid= np.linspace(- (max(betas_history) - min(betas_history)) - 1, (max(betas_history) - min(betas_history)) + 1, n_points)

	loss_surface= analyse_landscape_parallel(model, X, y, pca_1, pca_2, x_grid, y_grid)
	x_grid_, y_grid_ = np.meshgrid(x_grid, y_grid, indexing='ij')
	return loss_surface, x_grid_, y_grid_, alphas_history, betas_history, pca_1, pca_2, explained_var