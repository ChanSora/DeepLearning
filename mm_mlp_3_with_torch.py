import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import random
import time
# %matplotlib inline

class Tanh:
    def __call__(self, in_tensor):
        self.out = torch.tanh(in_tensor)
        return self.out
    def parameters(self):
        return []

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = (torch.randn((in_features, out_features))) / math.sqrt(in_features)
        self.weight.requires_grad_()
        self.bias = (torch.randn((1, out_features))) / math.sqrt(in_features) if bias == True else None
        self.bias.requires_grad_() if bias == True else None

    def __call__(self, x):
        self.out = x @ self.weight

        if self.bias is not None:
            self.out = self.out + self.bias

        return self.out

    def parameters(self):
        return (
            [self.weight, self.bias]
            if self.bias is not None else [self.weight]
        )

class Embedding:
    def __init__(self, vocab_size, emb_dim):
        self.Table = torch.randn((vocab_size, emb_dim))
        self.Table.requires_grad_()

    def __call__(self, x):
        self.out = self.Table[x]
        return self.out

    def parameters(self):
        return [self.Table]

class Flatten:

    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.training = True
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones((1, dim), requires_grad=True)
        self.beta = torch.zeros((1, dim), requires_grad=True)
        self.running_mean = torch.zeros((1, dim), requires_grad=True)
        self.running_var = torch.ones((1, dim), requires_grad=True)
    
    def __call__(self, x: torch.Tensor):
        
        if (self.training):
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        self.out = self.gamma * (x - xmean) / torch.sqrt(xvar + self.eps) + self.beta 

        if (self.training):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        with torch.no_grad():
            self.layers[-1].weight *= -1
            for layer in self.layers[:-1]:
                if isinstance(layer, Linear):
                    layer.weight *= 5/3

    def forward(self, x):
        self.out = x
        for layer in self.layers:
            self.out = layer(self.out)
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def build_dataset(words, stoi, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            idx = stoi[ch]
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

def plot_distributions(layers, step: int, bins: int = 100) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 4))

    for index, layer in enumerate(layers):
        if not isinstance(layer, torch.nn.Tanh):
            continue

        gradients = layer.out.grad
        if gradients is None:
            raise RuntimeError(
                f"layer {index}: 请先在前向传播后调用 retain_grad()，"
                "再执行 backward()"
            )

        activations = layer.out.detach().cpu().flatten()
        gradients = gradients.detach().cpu().flatten()
        label = f"layer {index}"

        # 激活值的概率密度直方图，各层使用相同的区间
        activation_density, activation_edges = torch.histogram(
            activations,
            bins=bins,
            range=(-1.0, 1.0),
            density=True,
        )
        activation_centers = (
            activation_edges[:-1] + activation_edges[1:]
        ) / 2

        axes[0].plot(
            activation_centers.numpy(),
            activation_density.numpy(),
            label=label,
        )

        # 梯度的概率密度直方图
        gradient_density, gradient_edges = torch.histogram(
            gradients,
            bins=bins,
            density=True,
        )
        gradient_centers = (
            gradient_edges[:-1] + gradient_edges[1:]
        ) / 2

        axes[1].plot(
            gradient_centers.numpy(),
            gradient_density.numpy(),
            label=label,
        )

        saturation = (activations.abs() > 0.97).float().mean().item()

        print(
            f"{label}: "
            f"activation mean={activations.mean().item():+.3f}, "
            f"std={activations.std().item():.3f}, "
            f"saturation={saturation:.1%}, "
            f"gradient std={gradients.std().item():.3e}"
        )

    axes[0].set_title(f"Activation distribution — step {step}")
    axes[1].set_title(f"Gradient distribution — step {step}")

    for ax in axes:
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.show()

def main():
    words = open('./makemore/names.txt', 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s, i in stoi.items()}
    vocab_size = len(stoi)
    block_size = 4
    n_embedded = 20
    n_hidden = 100
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, n_embedded),
            torch.nn.Flatten(),
            torch.nn.Linear(block_size * n_embedded, n_hidden, True), torch.nn.BatchNorm1d(n_hidden), 
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden, n_hidden, True), torch.nn.BatchNorm1d(n_hidden), 
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden, n_hidden, True), torch.nn.BatchNorm1d(n_hidden), 
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden, n_hidden, True), torch.nn.BatchNorm1d(n_hidden), 
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden, n_hidden, True), torch.nn.BatchNorm1d(n_hidden), 
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden, vocab_size, True),
        )

    model.to(device)

    def retain_module_output(module, inputs, output):
        module.out = output
        if output.requires_grad:
            output.retain_grad()

    for layer in model.modules():
        if layer is not model:
            layer.register_forward_hook(retain_module_output)

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1], stoi, block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)
    Xte, Yte = build_dataset(words[n2:], stoi, block_size)

    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xdev, Ydev = Xdev.to(device), Ydev.to(device)
    Xte, Yte = Xte.to(device), Yte.to(device)
    
    batch_size = 32
    max_steps = 100000
    lr = 0.1
    lossi = []

    torch.cuda.synchronize()
    start = time.time()
    for i in range(0, max_steps):
        # forward pass
        idx = torch.randint(0, Xtr.shape[0], (batch_size, ))
        x = Xtr[idx]

        logits = model.forward(x)

        loss = F.cross_entropy(logits, Ytr[idx])
        
        if (i % 10000 == 0):
            print(loss.item())

        # backward pass
        for p in model.parameters():
            p.grad = None 
        loss.backward()

        if (i == max_steps - 1):
            plot_distributions(model.modules(), max_steps)
        with torch.no_grad():
            for p in model.parameters():
                p.data += -lr * p.grad

    torch.cuda.synchronize()
    end = time.time()
    print(f"Training time: {end - start:.4f} seconds")

    split = {
            'train' : (Xtr, Ytr),
            'dev' : (Xdev, Ydev),
            'test' : (Xte, Yte)
        }

    with torch.no_grad():
        logits = model.forward(Xtr)
        train_loss = F.cross_entropy(logits, Ytr)
        print(f"train loss: {train_loss.item():.4f}")

    with torch.no_grad():
        logits = model.forward(Xdev)
        dev_loss = F.cross_entropy(logits, Ydev)
        print(f"dev loss: {dev_loss.item():.4f}")

if __name__ == "__main__":
    main()