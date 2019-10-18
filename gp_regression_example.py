import torch
import numpy as np
import matplotlib.pyplot as plt
import kernel as k
import gaussian_process as gp
from sklearn import ensemble
from sklearn import linear_model

dataset = np.genfromtxt('yacht_hydrodynamics.data')
obs = dataset.shape[0]
x = torch.from_numpy(dataset[:,:-1]).float()
y = torch.from_numpy(dataset[:,-1]).float().view(obs, 1)

order = np.random.permutation(obs)
split = 0.8
x_train = x[order[:int(split*obs)]]
y_train = y[order[:int(split*obs)]]
x_test = x[order[int(split*obs):]]
y_test = y[order[int(split*obs):]]


lin_model = linear_model.LinearRegression()
lin_model.fit(x_train.data.numpy(), y_train.flatten().data.numpy())
y_train_lin_hat = \
torch.from_numpy(lin_model.predict(x_train.data.numpy())).float().view(-1, 1)
y_test_lin_hat = \
torch.from_numpy(lin_model.predict(x_test.data.numpy())).float().view(-1, 1)

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
boost_model = ensemble.GradientBoostingRegressor(**params)
boost_model.fit(x_train.data.numpy(), y_train.flatten().data.numpy())
y_train_boost_hat = \
torch.from_numpy(boost_model.predict(x_train.data.numpy())).float().view(-1, 1)
y_test_boost_hat = \
torch.from_numpy(boost_model.predict(x_test.data.numpy())).float().view(-1, 1)

kernel = k.AdditiveMatern32Kernel(torch.ones(6), torch.ones(6))
model = gp.GaussianProcess(kernel, 0.01)
model.fit(x_train, y_train, lr=0.003)
with torch.no_grad():
    y_train_hat = model.predict(x_train)
    y_test_hat = model.predict(x_test)

kernel_list = [k.AdditiveMatern32Kernel(torch.ones(6), torch.ones(6)),
               k.AdditivePeriodicKernel(torch.ones(6), torch.ones(6),
                                        torch.ones(6))]
auto_model = gp.AutomaticGaussianProcess(kernel_list, None, 0.01)
auto_model.fit(x_train, y_train, lr=0.003, verbose=True)
with torch.no_grad():
    y_train_auto_hat = auto_model.predict(x_train)
    y_test_auto_hat = auto_model.predict(x_test)

lin_rmse_train = (y_train_lin_hat - y_train).pow(2).sum(-1).pow(0.5).mean()
lin_rmse_test = (y_test_lin_hat - y_test).pow(2).sum(-1).pow(0.5).mean()
boost_rmse_train = (y_train_boost_hat - y_train).pow(2).sum(-1).pow(0.5).mean()
boost_rmse_test = (y_test_boost_hat - y_test).pow(2).sum(-1).pow(0.5).mean()
rmse_train = (y_train_hat - y_train).pow(2).sum(-1).pow(0.5).mean()
rmse_test = (y_test_hat - y_test).pow(2).sum(-1).pow(0.5).mean()
auto_rmse_train = (y_train_auto_hat - y_train).pow(2).sum(-1).pow(0.5).mean()
auto_rmse_test = (y_test_auto_hat - y_test).pow(2).sum(-1).pow(0.5).mean()

t = np.linspace(y.min(), y.max(), 2)
fig, ax = plt.subplots(figsize=(10,5), nrows=1, ncols=2)
ax[0].plot(t, t, linestyle='--', alpha=0.5, c='k')
ax[0].plot(y_train.data.numpy(), y_train_lin_hat.data.numpy(),
           '.', c='g', alpha=0.5, label='Linear Model')
ax[0].plot(y_train.data.numpy(), y_train_boost_hat.data.numpy(),
           '.', c='dodgerblue', alpha=0.5, label='Gradient Boosting')
ax[0].plot(y_train.data.numpy(), y_train_hat.data.numpy(),
           '.', c='rebeccapurple', alpha=0.5, label='Gaussian Process')
ax[0].plot(y_train.data.numpy(), y_train_auto_hat.data.numpy(),
           '.', c='firebrick', alpha=0.5, label='Automatic Gaussian Process')
ax[0].set_xlim([y.min(), y.max()])
ax[0].set_ylim([y.min(), y.max()])
ax[0].set_xlabel('Target')
ax[0].set_ylabel('Prediction')
ax[0].set_title('Train')
ax[0].legend()
ax[1].plot(t, t, linestyle='--', alpha=0.5, c='k')
ax[1].plot(y_test.data.numpy(), y_test_lin_hat.data.numpy(),
           '.', c='g', alpha=0.5, label='Linear Model')
ax[1].plot(y_test.data.numpy(), y_test_boost_hat.data.numpy(),
           '.', c='dodgerblue', alpha=0.5, label='Gradient Boosting')
ax[1].plot(y_test.data.numpy(), y_test_hat.data.numpy(),
           '.', c='rebeccapurple', alpha=0.5, label='Gaussian Process')
ax[1].plot(y_test.data.numpy(), y_test_auto_hat.data.numpy(),
           '.', c='firebrick', alpha=0.5, label='Automatic Gaussian Process')
ax[1].set_xlim([y.min(), y.max()])
ax[1].set_ylim([y.min(), y.max()])
ax[1].set_xlabel('Target')
ax[1].set_ylabel('Prediction')
ax[1].set_title('Test')
ax[1].legend()
plt.show()
print('Linear Train RMSE =', lin_rmse_train.item())
print('Linear Test RMSE =', lin_rmse_test.item())
print('Boost Train RMSE =', boost_rmse_train.item())
print('Boost Test RMSE =', boost_rmse_test.item())
print('GP Train RMSE =', rmse_train.item())
print('GP Test RMSE =', rmse_test.item())
print('Auto Train RMSE =', auto_rmse_train.item())
print('Auto Test RMSE =', auto_rmse_test.item())